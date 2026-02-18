"""
Orchestrates: segments → optional CV (tracking) → VLM analysis → optional OCR → chunk assembly → JSONL.

CV pipeline can run before or in parallel to VLM when enable_cv=True.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import (
    Segment,
    Tracklet,
    VideoChunk,
    build_chunk_hash,
    compose_index_text,
)
from .video_processor import segment_video
from .vlm_backend import VLMBackend, unpack_vlm_for_log
from .tracker import NoopTracker, SimpleColorBlobTracker, Tracker, YOLOTracker, YOLOAPITracker
from .deblurrer import Deblurrer, _get_deblurrer

logger = logging.getLogger(__name__)


def _get_tracker(object_tracker: str | None, enable_cv: bool = False) -> Tracker:
    """Select tracker from config string. When enable_cv=False, always use NoopTracker."""
    if not enable_cv:
        return NoopTracker()
    if object_tracker == "simple_blob":
        return SimpleColorBlobTracker()
    if object_tracker == "yolo_api":
        try:
            from shared.config import settings
            url = getattr(settings, "YOLO_API_URL", None) or ""
            key = getattr(settings, "YOLO_API_KEY", "") or ""
            model = getattr(settings, "YOLO_MODEL_ID", "yolov8n-640/1") or "yolov8n-640/1"
            if not url or not key:
                logger.warning("YOLO_API_URL and YOLO_API_KEY required for yolo_api tracker; using NoopTracker")
                return NoopTracker()
            return YOLOAPITracker(api_url=url, api_key=key, model_id=model)
        except Exception as e:
            logger.warning("Failed to create YOLOAPITracker: %s; using NoopTracker", e)
            return NoopTracker()
    if object_tracker in ("yolo", "yolo11", None):
        try:
            from shared.config import settings
            url = getattr(settings, "YOLO_API_URL", None) or ""
            key = getattr(settings, "YOLO_API_KEY", "") or ""
            model = getattr(settings, "YOLO_MODEL_ID", "yolov8n-640/1") or "yolov8n-640/1"
            if url and key:
                return YOLOAPITracker(api_url=url, api_key=key, model_id=model)
        except Exception as e:
            logger.debug("YOLO API not configured: %s; using local YOLOTracker", e)
        return YOLOTracker()
    return NoopTracker()


def _tracks_to_json(tracks: list[Tracklet]) -> str:
    """Serialize tracks to JSON. Returns '[]' when empty so cv_meta is always present."""
    return json.dumps([t.model_dump() for t in tracks], ensure_ascii=False)


_paddle_ocr_instance: Any = None


OCR_MODEL_NAME = "PaddleOCR"


def _parse_ocr_max_sides() -> list[int]:
    from shared.config import settings
    raw = getattr(settings, "OCR_MAX_SIDES", "2200,3000")
    max_sides: list[int] = []
    for s in str(raw).split(","):
        s = s.strip()
        if not s:
            continue
        try:
            max_sides.append(int(s))
        except ValueError:
            continue
    if not max_sides:
        max_sides = [2200, 3000]
    return max_sides


def _get_paddle_ocr():
    """Lazy-load PaddleOCR (cached). Returns None if not installed."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is not None:
        return _paddle_ocr_instance
    try:
        from paddleocr import PaddleOCR
        from shared.config import settings

        _paddle_ocr_instance = PaddleOCR(
            use_angle_cls=getattr(settings, "OCR_USE_ANGLE_CLS", True),
            lang="en",
            show_log=False,
            det_db_thresh=getattr(settings, "OCR_DET_DB_THRESH", 0.25),
            det_db_box_thresh=getattr(settings, "OCR_DET_DB_BOX_THRESH", 0.55),
            det_limit_side_len=getattr(settings, "OCR_DET_LIMIT_SIDE_LEN", 1920),
            det_limit_type=getattr(settings, "OCR_DET_LIMIT_TYPE", "max"),
        )
        return _paddle_ocr_instance
    except ImportError as e:
        logger.warning("PaddleOCR not installed, skipping OCR: %s", e)
        return None


def _run_ocr_on_segment(seg: Segment, frames_override: list[Any] | None = None) -> str:
    """Extract text from segment frames using multi-scale PaddleOCR."""
    from shared.config import settings
    from .ocr_processor import ocr_single_image

    ocr = _get_paddle_ocr()
    if ocr is None:
        return ""

    max_sides = _parse_ocr_max_sides()
    min_conf = getattr(settings, "OCR_MIN_CONF", 0.55)
    merge_iou = getattr(settings, "OCR_MERGE_IOU", 0.35)
    keep_topk = getattr(settings, "OCR_KEEP_TOPK", 200)
    use_angle_cls = getattr(settings, "OCR_USE_ANGLE_CLS", True)
    use_clahe = getattr(settings, "OCR_USE_CLAHE", True)
    clahe_clip_limit = getattr(settings, "OCR_CLAHE_CLIP_LIMIT", 2.5)
    unsharp_sigma = getattr(settings, "OCR_UNSHARP_SIGMA", 1.1)
    unsharp_amount = getattr(settings, "OCR_UNSHARP_AMOUNT", 0.9)

    parts: list[str] = []
    frames = frames_override if frames_override is not None else seg.frames
    for i, (frame, t) in enumerate(zip(frames, seg.frame_times)):
        if frame is None or not isinstance(frame, np.ndarray):
            continue
        try:
            text = ocr_single_image(
                bgr=frame,
                ocr=ocr,
                max_sides=max_sides,
                min_conf=min_conf,
                merge_iou=merge_iou,
                keep_topk=keep_topk,
                use_angle_cls=use_angle_cls,
                use_clahe=use_clahe,
                clahe_clip_limit=clahe_clip_limit,
                unsharp_sigma=unsharp_sigma,
                unsharp_amount=unsharp_amount,
            )
            parts.append(f"[t={t:.1f}s] {text}" if text else f"[t={t:.1f}s] (none)")
        except Exception as e:
            logger.debug("OCR failed for frame %d: %s", i, e)
            parts.append(f"[t={t:.1f}s] (error: {e})")
    return "\n".join(parts) if parts else ""


def _ocr_text_for_embedding(ocr_text: str) -> str:
    """Filter OCR output to only include lines where text was detected."""
    if not ocr_text:
        return ""
    lines = [l for l in ocr_text.split("\n") if "(none)" not in l and "(error:" not in l]
    return "\n".join(lines) if lines else ""


def _assemble_chunk(
    seg: Segment,
    vlm_out: dict[str, Any],
    tracks: list[Tracklet],
    ocr_text: str | None = None,
) -> VideoChunk:
    if ocr_text:
        vlm_out = {**vlm_out, "ocr_text": ocr_text}
    index_text = compose_index_text(
        vlm_out.get("caption", ""),
        vlm_out.get("events", []),
        vlm_out.get("entities", []),
        None,  # transcript_slice stub
        ocr_text=_ocr_text_for_embedding(ocr_text) if ocr_text else None,
    )
    h = build_chunk_hash(
        seg.video_id,
        seg.t_start,
        seg.t_end,
        seg.frame_times,
        vlm_out.get("model_id", ""),
        vlm_out.get("prompt_id", ""),
    )
    cv_meta = _tracks_to_json(tracks)
    return VideoChunk(
        video_id=seg.video_id,
        t_start=seg.t_start,
        t_end=seg.t_end,
        index_text=index_text,
        fields=vlm_out,
        frame_times=seg.frame_times,
        tracks=tracks,
        cv_meta=cv_meta,
        hash=h,
    )


def _detections_for_frame(
    frame_t: float,
    tracks: list[Tracklet],
    tolerance_sec: float = 0.05,
) -> list[dict[str, Any]]:
    """Extract detections at a given frame timestamp from tracklets."""
    detections: list[dict[str, Any]] = []
    for tr in tracks:
        for bbox in tr.bboxes:
            if abs(bbox.t - frame_t) <= tolerance_sec:
                detections.append({
                    "track_id": tr.track_id,
                    "label": tr.label,
                    "bbox": {"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h},
                })
                break
    return detections


def _draw_detections_on_frame(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame. Returns annotated image."""
    img = frame.copy()
    for det in detections:
        bbox = det.get("bbox", {})
        x, y = bbox.get("x", 0), bbox.get("y", 0)
        w, h = bbox.get("w", 0), bbox.get("h", 0)
        label = det.get("label") or det.get("track_id", "?")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 4, y), (0, 255, 0), -1)
        cv2.putText(img, str(label), (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


def _write_cv_debug_output(
    debug_dir: Path,
    video_id: str,
    model_name: str,
    seg_idx: int,
    frames: list[Any],
    frame_times: list[float],
    tracks: list[Tracklet],
) -> None:
    """Write per-frame images and detections JSON to disk for CV pipeline debug."""
    seg_dir = debug_dir / f"{video_id}_{model_name}" / f"seg_{seg_idx:04d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "CV debug: writing %d frames to %s",
        len(frames),
        seg_dir,
    )
    for i, (frame, t_sec) in enumerate(zip(frames, frame_times)):
        if frame is None or not isinstance(frame, np.ndarray):
            continue
        img_path = seg_dir / f"frame_{i:04d}_t{t_sec:.2f}s.png"
        cv2.imwrite(str(img_path), frame)
        detections = _detections_for_frame(t_sec, tracks)
        det_path = seg_dir / f"frame_{i:04d}_t{t_sec:.2f}s_detections.json"
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(
                {"t": t_sec, "frame_idx": i, "detections": detections},
                f,
                indent=2,
                ensure_ascii=False,
            )
        if detections:
            annotated = _draw_detections_on_frame(frame, detections)
            annotated_path = seg_dir / f"frame_{i:04d}_t{t_sec:.2f}s_boxes.png"
            cv2.imwrite(str(annotated_path), annotated)
    logger.debug(
        "CV debug: wrote segment %d (%d frames) to %s",
        seg_idx,
        len([f for f in frames if f is not None]),
        seg_dir,
    )


def _truncate_for_log(obj: Any, caption_max: int = 500, events_max: int = 5) -> dict[str, Any]:
    """Shrink vlm_out for logging to avoid huge log lines."""
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "caption" in out and isinstance(out["caption"], str) and len(out["caption"]) > caption_max:
        out["caption"] = out["caption"][:caption_max] + "..."
    if "events" in out and isinstance(out["events"], list) and len(out["events"]) > events_max:
        out["events"] = out["events"][:events_max] + [f"... and {len(out['events']) - events_max} more"]
    return out


def run_video_pipeline(
    path: str | Path,
    out_path: str | Path,
    *,
    video_id: str | None = None,
    mode: str = "procedure",
    segment_sec: float = 4.0,
    overlap_sec: float = 1.0,
    num_frames: int = 10,
    vlm: VLMBackend | None = None,
    tracker: Tracker | None = None,
    object_tracker: str | None = None,
    enable_cv: bool = False,
    cv_before_vlm: bool = True,
    cv_parallel: bool = False,
    vlm_prompt: str | None = None,
    enable_ocr: bool = False,
    log_vlm_outcome: bool = False,
    log_vlm_outcome_sample_every: int = 1,
    log_vlm_review_file: bool = False,
    cv_debug_output_dir: str | Path | None = None,
    deblurrer: Deblurrer | None = None,
    deblurrer_name: str = "none",
) -> dict[str, Any]:
    """
    Ingest MP4: segment → [CV before/parallel] → VLM → chunks → JSONL.

    When enable_cv=True:
      - cv_before_vlm=True: Run CV on all segments first, then VLM (with CV results).
      - cv_before_vlm=False: Run CV and VLM sequentially per segment (legacy).
      - cv_parallel=True (with cv_before_vlm): Run CV on segments in parallel.

    Returns summary: num_segments, avg_frames_per_segment, elapsed_sec.
    """
    from .vlm_backend import DummyVLMBackend

    path = Path(path)
    out_path = Path(out_path)
    vlm = vlm or DummyVLMBackend()
    tracker = tracker or _get_tracker(object_tracker, enable_cv=enable_cv)
    deblurrer = deblurrer or _get_deblurrer(deblurrer_name)
    v_id = video_id or path.stem
    debug_dir = Path(cv_debug_output_dir) if cv_debug_output_dir else None

    logger.info(
        "CV pipeline config: enable_cv=%s object_tracker=%s cv_before_vlm=%s cv_parallel=%s debug_output=%s",
        enable_cv,
        object_tracker,
        cv_before_vlm,
        cv_parallel,
        str(debug_dir) if debug_dir else None,
    )

    t0 = time.perf_counter()
    segments = segment_video(
        path,
        video_id=v_id,
        segment_sec=segment_sec,
        overlap_sec=overlap_sec,
        num_frames=num_frames,
    )
    seg_elapsed = time.perf_counter() - t0
    logger.info("Segmentation: %d segments in %.2fs", len(segments), seg_elapsed)

    # Deblur frames for each segment (before CV and VLM)
    deblurred_frames_list = [deblurrer.deblur_frames(seg.frames) for seg in segments]

    # Run CV before VLM when enable_cv and cv_before_vlm
    cv_results: list[tuple[list[Tracklet], str]] = []
    if enable_cv and cv_before_vlm:
        cv_start = time.perf_counter()
        logger.info("CV pipeline: starting tracking on %d segments (parallel=%s)", len(segments), cv_parallel)
        if cv_parallel:
            with ThreadPoolExecutor(max_workers=min(4, len(segments) or 1)) as ex:
                futures = {
                    ex.submit(tracker.tracks_for_segment, deblurred_frames_list[seg_idx], seg.frame_times): seg_idx
                    for seg_idx, seg in enumerate(segments)
                }
                results_by_idx: dict[int, tuple[list[Tracklet], str]] = {}
                for fut in as_completed(futures):
                    seg_idx = futures[fut]
                    tracks = fut.result()
                    tracks_json = _tracks_to_json(tracks)
                    results_by_idx[seg_idx] = (tracks, tracks_json)
                    logger.debug("CV pipeline: segment %d done, %d tracks", seg_idx, len(tracks))
                cv_results = [results_by_idx[i] for i in range(len(segments))]
        else:
            for seg_idx, seg in enumerate(segments):
                logger.debug("CV pipeline: processing segment %d (t=%.1f-%.1fs, %d frames)", seg_idx, seg.t_start, seg.t_end, len(seg.frame_times))
                tracks = tracker.tracks_for_segment(deblurred_frames_list[seg_idx], seg.frame_times)
                cv_results.append((tracks, _tracks_to_json(tracks)))
        cv_elapsed = time.perf_counter() - cv_start
        total_tracks = sum(len(t) for t, _ in cv_results)
        logger.info("CV pipeline: %d segments in %.2fs, %d total tracks", len(segments), cv_elapsed, total_tracks)

    total_frames = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    review_path = Path(str(out_path) + ".vlm_review.jsonl") if log_vlm_review_file else None
    review_file = None
    if review_path and log_vlm_outcome:
        review_file = open(review_path, "w", encoding="utf-8")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for seg_idx, seg in enumerate(segments):
                frames_to_use = deblurred_frames_list[seg_idx]
                if enable_cv and cv_before_vlm and cv_results:
                    tracks, tracks_json = cv_results[seg_idx]
                    logger.debug("CV pipeline: segment %d using pre-computed tracks (%d)", seg_idx, len(tracks))
                else:
                    logger.debug("CV pipeline: segment %d running tracker (%d frames)", seg_idx, len(frames_to_use))
                    tracks = tracker.tracks_for_segment(frames_to_use, seg.frame_times)
                    tracks_json = _tracks_to_json(tracks)
                if debug_dir:
                    model_name = getattr(tracker, "model_name", "unknown")
                    _write_cv_debug_output(
                        debug_dir,
                        v_id,
                        model_name,
                        seg_idx,
                        frames_to_use,
                        seg.frame_times,
                        tracks,
                    )
                if tracks:
                    logger.info(
                        "Tracking output video_id=%s t_start=%.1f t_end=%.1f segment_index=%d num_tracks=%d\n%s",
                        seg.video_id,
                        seg.t_start,
                        seg.t_end,
                        seg_idx,
                        len(tracks),
                        json.dumps([t.model_dump() for t in tracks], indent=2, ensure_ascii=False),
                    )

                ocr_text = _run_ocr_on_segment(seg, frames_override=frames_to_use) if enable_ocr else None
                if enable_ocr and ocr_text:
                    logger.info(
                        "OCR output video_id=%s t_start=%.1f t_end=%.1f segment_index=%d\n%s",
                        seg.video_id,
                        seg.t_start,
                        seg.t_end,
                        seg_idx,
                        ocr_text,
                    )
                if debug_dir and enable_ocr:
                    model_name = getattr(tracker, "model_name", "unknown")
                    seg_dir = debug_dir / f"{v_id}_{model_name}" / f"seg_{seg_idx:04d}"
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        from shared.config import settings
                        ocr_opts = (
                            f"max_sides={getattr(settings, 'OCR_MAX_SIDES', '2200,3000')}, "
                            f"min_conf={getattr(settings, 'OCR_MIN_CONF', 0.55)}, "
                            f"merge_iou={getattr(settings, 'OCR_MERGE_IOU', 0.35)}, "
                            f"clahe={getattr(settings, 'OCR_USE_CLAHE', True)}, "
                            f"angle_cls={getattr(settings, 'OCR_USE_ANGLE_CLS', True)}"
                        )
                    except Exception:
                        ocr_opts = ""
                    ocr_content = f"OCR model: {OCR_MODEL_NAME}\nOCR options: {ocr_opts}\n\n{ocr_text or ''}"
                    (seg_dir / "ocr_text.txt").write_text(ocr_content, encoding="utf-8")

                vlm_out = vlm.analyze_segment(
                    video_id=seg.video_id,
                    t_start=seg.t_start,
                    t_end=seg.t_end,
                    frame_times=seg.frame_times,
                    frames_bgr=frames_to_use,
                    mode=mode,
                    tracks=tracks if tracks else None,
                    extra_context=tracks_json if tracks else None,
                    custom_prompt=vlm_prompt,
                )

                if log_vlm_outcome and (seg_idx % log_vlm_outcome_sample_every == 0):
                    unpacked = unpack_vlm_for_log(vlm_out)
                    unpacked = _truncate_for_log(unpacked)
                    logger.info(
                        "VLM outcome video_id=%s t_start=%.1f t_end=%.1f segment_index=%d\n%s",
                        seg.video_id,
                        seg.t_start,
                        seg.t_end,
                        seg_idx,
                        json.dumps(unpacked, indent=2, ensure_ascii=False),
                    )
                    if review_file is not None:
                        review_file.write(
                            json.dumps(
                                {
                                    "video_id": seg.video_id,
                                    "t_start": seg.t_start,
                                    "t_end": seg.t_end,
                                    "segment_index": seg_idx,
                                    "vlm_out": vlm_out,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                chunk = _assemble_chunk(seg, vlm_out, tracks, ocr_text=ocr_text)
                total_frames += len(seg.frame_times)
                f.write(chunk.model_dump_json() + "\n")
    finally:
        if review_file is not None:
            review_file.close()

    elapsed = time.perf_counter() - t0
    n = len(segments)
    avg_frames = (total_frames / n) if n else 0

    return {
        "num_segments": n,
        "avg_frames_per_segment": round(avg_frames, 1),
        "elapsed_sec": round(elapsed, 2),
        "out_path": str(out_path),
    }

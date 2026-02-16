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


def _tracks_to_json(tracks: list[Tracklet]) -> str | None:
    if not tracks:
        return None
    return json.dumps([t.model_dump() for t in tracks], ensure_ascii=False)


def _run_ocr_on_segment(seg: Segment, frames_override: list[Any] | None = None) -> str:
    """Extract text from segment frames using Tesseract OCR."""
    parts: list[str] = []
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed, skipping OCR")
        return ""

    frames = frames_override if frames_override is not None else seg.frames
    for i, (frame, t) in enumerate(zip(frames, seg.frame_times)):
        if frame is None or not isinstance(frame, np.ndarray):
            continue
        try:
            # BGR to RGB for pytesseract
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(rgb).strip()
            if text:
                parts.append(f"[t={t:.1f}s] {text}")
        except Exception as e:
            logger.debug("OCR failed for frame %d: %s", i, e)
    return "\n".join(parts) if parts else ""


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
        ocr_text=ocr_text or None,
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
    cv_results: list[tuple[list[Tracklet], str | None]] = []
    if enable_cv and cv_before_vlm:
        cv_start = time.perf_counter()
        if cv_parallel:
            with ThreadPoolExecutor(max_workers=min(4, len(segments) or 1)) as ex:
                futures = {
                    ex.submit(tracker.tracks_for_segment, deblurred_frames_list[seg_idx], seg.frame_times): seg_idx
                    for seg_idx, seg in enumerate(segments)
                }
                results_by_idx: dict[int, tuple[list[Tracklet], str | None]] = {}
                for fut in as_completed(futures):
                    seg_idx = futures[fut]
                    tracks = fut.result()
                    tracks_json = _tracks_to_json(tracks)
                    results_by_idx[seg_idx] = (tracks, tracks_json)
                cv_results = [results_by_idx[i] for i in range(len(segments))]
        else:
            for seg_idx, seg in enumerate(segments):
                tracks = tracker.tracks_for_segment(deblurred_frames_list[seg_idx], seg.frame_times)
                cv_results.append((tracks, _tracks_to_json(tracks)))
        cv_elapsed = time.perf_counter() - cv_start
        logger.info("CV pipeline: %d segments in %.2fs", len(segments), cv_elapsed)

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
                else:
                    tracks = tracker.tracks_for_segment(frames_to_use, seg.frame_times)
                    tracks_json = _tracks_to_json(tracks)
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

                vlm_out = vlm.analyze_segment(
                    video_id=seg.video_id,
                    t_start=seg.t_start,
                    t_end=seg.t_end,
                    frame_times=seg.frame_times,
                    frames_bgr=frames_to_use,
                    mode=mode,
                    tracks=tracks if tracks else None,
                    extra_context=tracks_json,
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

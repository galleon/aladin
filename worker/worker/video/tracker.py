"""
Tracker protocol and implementations: NoopTracker, SimpleColorBlobTracker, YOLOTracker.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

import cv2
import numpy as np

from .schemas import BBoxFrame, Tracklet


def _load_yolo():
    """Lazy load Ultralytics YOLO. Raises ImportError with helpful message if not installed."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for YOLOTracker. Install with: pip install ultralytics"
        ) from e


class Tracker(Protocol):
    def tracks_for_segment(self, frames_bgr: list[Any], frame_times: list[float]) -> list[Tracklet]:
        """Produce tracklets for the segment. frames_bgr: list[ndarray|None] (BGR)."""
        ...


class NoopTracker:
    """Returns empty tracks."""

    def tracks_for_segment(self, frames_bgr: list[Any], frame_times: list[float]) -> list[Tracklet]:
        return []


class SimpleColorBlobTracker:
    """
    MVP: detect moving blobs via frame-diff, pick largest N, assign stable fake IDs
    (T01, T02, ...) and bboxes per frame. Not accuracy-focused; validates the plumbing.
    """

    def __init__(self, max_tracks: int = 5, min_area: int = 500):
        self.max_tracks = max(1, max_tracks)
        self.min_area = min_area

    def tracks_for_segment(self, frames_bgr: list[Any], frame_times: list[float]) -> list[Tracklet]:
        valid = [(f, t) for f, t in zip(frames_bgr, frame_times) if f is not None and isinstance(f, np.ndarray)]
        if len(valid) < 2:
            return []

        # Build simple "moving" regions: |frame_i - frame_{i-1}|, threshold, contours
        h, w = valid[0][0].shape[:2]
        # Accumulate motion over pairs; then for each frame assign bboxes to "tracks" by position overlap
        all_boxes: list[list[tuple[int, int, int, int]]] = []  # per-frame list of (x,y,w,h)

        for i in range(len(valid)):
            curr = cv2.cvtColor(valid[i][0], cv2.COLOR_BGR2GRAY)
            if i == 0:
                prev = curr
            diff = cv2.absdiff(prev, curr)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            th = cv2.dilate(th, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes: list[tuple[int, int, int, int]] = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((int(x), int(y), int(bw), int(bh)))
            # Sort by area desc, take up to max_tracks
            boxes.sort(key=lambda r: r[2] * r[3], reverse=True)
            all_boxes.append(boxes[: self.max_tracks])
            prev = curr

        # Assign stable IDs by greedy matching: for each frame, match boxes to "tracks" by bbox center proximity
        # Simplified: assume order is stable (largest blob = T01, second = T02, ...) and fill bboxes per track
        track_bboxes: dict[int, list[BBoxFrame]] = {j: [] for j in range(self.max_tracks)}

        for idx, (boxes, (_, t)) in enumerate(zip(all_boxes, valid)):
            for j, (x, y, bw, bh) in enumerate(boxes):
                if j < self.max_tracks:
                    track_bboxes[j].append(BBoxFrame(t=t, x=x, y=y, w=bw, h=bh))

        out: list[Tracklet] = []
        for j in range(self.max_tracks):
            bboxes = track_bboxes[j]
            if not bboxes:
                continue
            track_id = f"T{j + 1:02d}"
            out.append(Tracklet(track_id=track_id, bboxes=bboxes, label="blob"))
        return out


class YOLOTracker:
    """
    YOLOv11 Tracker implementation using Ultralytics.
    Maps detection results to the VSS Tracklet schema.
    Model downloads on first use (e.g. yolo11n.pt).
    """

    def __init__(self, model_path: str = "yolo11n.pt"):
        YOLO = _load_yolo()
        self.model = YOLO(model_path)

    def tracks_for_segment(self, frames_bgr: list[Any], frame_times: list[float]) -> list[Tracklet]:
        """
        Run YOLO tracking on a batch of frames and convert to Tracklets.
        """
        valid = [
            (f, t)
            for f, t in zip(frames_bgr, frame_times)
            if f is not None and isinstance(f, np.ndarray)
        ]
        if not valid:
            return []

        valid_frames = [v[0] for v in valid]
        valid_times = [v[1] for v in valid]

        results = self.model.track(valid_frames, persist=True, verbose=False)

        tracklets_map: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"bboxes": {}, "label": "object"}
        )

        for i, result in enumerate(results):
            timestamp = valid_times[i]

            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            class_ids = result.boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                cx, cy, w, h = box
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                label = result.names[int(cls_id)]
                tid = str(int(track_id))

                tracklets_map[tid]["bboxes"][timestamp] = BBoxFrame(
                    t=timestamp, x=max(0, x), y=max(0, y), w=max(1, int(w)), h=max(1, int(h))
                )
                tracklets_map[tid]["label"] = label

        tracklets: list[Tracklet] = []
        for tid, data in tracklets_map.items():
            bboxes_dict = data["bboxes"]
            if not bboxes_dict:
                continue
            bboxes_list = sorted(bboxes_dict.values(), key=lambda b: b.t)
            tracklets.append(
                Tracklet(
                    track_id=tid,
                    bboxes=bboxes_list,
                    label=data["label"],
                )
            )
        return tracklets


def _load_yolo_api_client(api_url: str, api_key: str):
    """Lazy load InferenceHTTPClient. Raises ImportError with helpful message if not installed."""
    try:
        from inference_sdk import InferenceHTTPClient
        return InferenceHTTPClient(api_url=api_url.rstrip("/"), api_key=api_key)
    except ImportError as e:
        raise ImportError(
            "inference-sdk is required for YOLOAPITracker. Install with: pip install inference-sdk"
        ) from e


def _load_byte_tracker():
    """Lazy load ByteTrack from supervision."""
    try:
        from supervision import ByteTrack
        return ByteTrack
    except ImportError as e:
        raise ImportError(
            "supervision is required for YOLOAPITracker. Install with: pip install supervision"
        ) from e


def _load_supervision_detections():
    """Lazy load Detections from supervision."""
    try:
        from supervision import Detections
        return Detections
    except ImportError as e:
        raise ImportError(
            "supervision is required for YOLOAPITracker. Install with: pip install supervision"
        ) from e


class YOLOAPITracker:
    """
    YOLO Tracker using Roboflow Inference API (hosted or self-hosted).
    Per-frame detection via API + client-side ByteTrack for persistent track IDs.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_id: str,
        confidence: float = 0.25,
    ):
        self.client = _load_yolo_api_client(api_url, api_key)
        self.model_id = model_id
        self.confidence = confidence
        ByteTrack = _load_byte_tracker()
        self.tracker = ByteTrack()
        self.Detections = _load_supervision_detections()

    def tracks_for_segment(self, frames_bgr: list[Any], frame_times: list[float]) -> list[Tracklet]:
        """
        Run detection via API per frame, ByteTrack for IDs, convert to Tracklets.
        """
        valid = [
            (f, t)
            for f, t in zip(frames_bgr, frame_times)
            if f is not None and isinstance(f, np.ndarray)
        ]
        if not valid:
            return []

        tracklets_map: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"bboxes": {}, "label": "object"}
        )

        infer_config = None
        try:
            from inference_sdk import InferenceConfiguration
            infer_config = InferenceConfiguration(confidence_threshold=self.confidence)
        except ImportError:
            pass

        for frame_bgr, timestamp in valid:
            try:
                if infer_config is not None:
                    with self.client.use_configuration(infer_config):
                        result = self.client.infer(frame_bgr, model_id=self.model_id)
                else:
                    result = self.client.infer(frame_bgr, model_id=self.model_id)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("YOLO API inference failed for frame at %.1fs: %s", timestamp, e)
                continue

            predictions = result.get("predictions") or []
            if not predictions:
                self.tracker.update_with_detections(self.Detections.empty())
                continue

            detections = self.Detections.from_inference(result)
            tracked = self.tracker.update_with_detections(detections)

            if tracked.tracker_id is None:
                continue

            labels = [p.get("class", "object") for p in predictions]
            for i, tid in enumerate(tracked.tracker_id):
                if tid is None:
                    continue
                tid_int = int(tid)
                xyxy = tracked.xyxy[i]
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                x = max(0, int(x1))
                y = max(0, int(y1))
                w = max(1, int(x2 - x1))
                h = max(1, int(y2 - y1))
                label = labels[i] if i < len(labels) else "object"
                tracklets_map[tid_int]["bboxes"][timestamp] = BBoxFrame(t=timestamp, x=x, y=y, w=w, h=h)
                tracklets_map[tid_int]["label"] = label

        tracklets: list[Tracklet] = []
        for tid, data in tracklets_map.items():
            bboxes_dict = data["bboxes"]
            if not bboxes_dict:
                continue
            bboxes_list = sorted(bboxes_dict.values(), key=lambda b: b.t)
            tracklets.append(
                Tracklet(
                    track_id=str(tid),
                    bboxes=bboxes_list,
                    label=data["label"],
                )
            )
        return tracklets

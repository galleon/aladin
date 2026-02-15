"""
Video segmentation and frame sampling. Fixed windows with biased_edges strategy.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .schemas import Segment

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_SEGMENT_SEC = 4.0
DEFAULT_OVERLAP_SEC = 1.0
DEFAULT_NUM_FRAMES = 10
MIN_FRAMES = 8
MAX_FRAMES = 12


def _get_duration(cap: cv2.VideoCapture) -> float | None:
    """Return duration in seconds if CAP_PROP_FRAME_COUNT and FPS are valid, else None."""
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if count is None or count <= 0 or fps is None or fps <= 0:
        return None
    return float(count) / float(fps)


def _get_duration_by_iteration(cap: cv2.VideoCapture) -> float:
    """
    Fallback when CAP_PROP_FRAME_COUNT is missing: iterate until read fails.
    Resets cap to start. Returns duration in seconds.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = 0
    while True:
        ok = cap.read()[0]
        if not ok:
            break
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return (n / float(fps)) if n else 0.0


def biased_edges_timestamps(
    t_start: float,
    t_end: float,
    n: int,
) -> list[float]:
    """
    Generate n timestamps in [t_start, t_end] with more weight near edges.
    - A couple near start (after t_start)
    - A couple near end (before t_end)
    - Rest spread uniformly in the middle.
    """
    if n <= 0:
        return []
    if t_end <= t_start:
        return []

    span = t_end - t_start
    # Edge bias: 2 near start, 2 near end; rest in middle
    n_start = min(2, max(0, n // 4))
    n_end = min(2, max(0, n // 4))
    n_mid = max(0, n - n_start - n_end)

    out: list[float] = []
    # Near start: 10% and 20% into the segment
    for i in range(n_start):
        frac = (i + 1) / (n_start + 1) * 0.25  # within first quarter
        out.append(t_start + span * frac)
    # Middle: uniform
    for i in range(n_mid):
        frac = (i + 1) / (n_mid + 1)  # (0,1) over the "middle" portion
        # map to [0.25, 0.75] of segment
        out.append(t_start + span * (0.25 + 0.5 * frac))
    # Near end: 80% and 90% into the segment
    for i in range(n_end):
        frac = 0.75 + (i + 1) / (n_end + 1) * 0.25
        out.append(t_start + span * frac)

    out.sort()
    return out


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float, fps: float) -> np.ndarray | None:
    """Seek to t_sec and read one frame. Returns BGR or None on failure."""
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    return None


def segment_video(
    path: str | Path,
    video_id: str | None = None,
    *,
    segment_sec: float = DEFAULT_SEGMENT_SEC,
    overlap_sec: float = DEFAULT_OVERLAP_SEC,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> list[Segment]:
    """
    Split video into fixed-window segments and sample frames with biased_edges.
    Each Segment holds t_start, t_end, frame_times, and frames (BGR arrays; None on read failure).

    - If CAP_PROP_FRAME_COUNT is missing/invalid, duration is computed by iterating until read fails.
    - Failed frame reads near the end are stored as None; the segment is still yielded.
    """
    path = Path(path)
    vid = str(path.resolve())
    v_id = video_id if video_id is not None else path.stem

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        duration = _get_duration(cap)
        if duration is None:
            duration = _get_duration_by_iteration(cap)
            logger.debug("Duration from iteration: %.2f s", duration)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = max(MIN_FRAMES, min(MAX_FRAMES, num_frames))
        step = max(0.0, segment_sec - overlap_sec)
        if step <= 0:
            step = segment_sec

        out: list[Segment] = []
        t = 0.0
        while t < duration:
            t_end = min(t + segment_sec, duration)
            if t >= t_end:
                break

            frame_times = biased_edges_timestamps(t, t_end, n_frames)
            frames: list[np.ndarray | None] = []
            for ts in frame_times:
                f = _read_frame_at(cap, ts, fps)
                frames.append(f)

            out.append(
                Segment(
                    video_id=v_id,
                    t_start=t,
                    t_end=t_end,
                    frame_times=frame_times,
                    frames=frames,
                )
            )
            t += step
            if t >= duration:
                break

        return out
    finally:
        cap.release()

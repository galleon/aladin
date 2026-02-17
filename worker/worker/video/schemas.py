"""
Pydantic models and dataclasses for video ingestion: Segment, Tracklet, VideoChunk, and VLM I/O.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


# --- VLM response (from backend) ---


class VLMResponse(BaseModel):
    """Structured response from VLMBackend.analyze_segment."""

    caption: str = Field(default="", description="Scene-level summary.")
    events: list[dict[str, Any]] = Field(default_factory=list, description="List of events.")
    entities: list[dict[str, Any]] = Field(default_factory=list, description="List of entities.")
    notes: list[str] = Field(default_factory=list, description="Additional notes.")
    model_id: str = Field(default="", description="VLM model identifier for caching.")
    prompt_id: str = Field(default="", description="Prompt template version for caching.")


# --- Tracker output ---


class BBoxFrame(BaseModel):
    """Bounding box for one frame."""

    t: float = Field(..., description="Timestamp in seconds.")
    x: int = Field(..., ge=0, description="Left.")
    y: int = Field(..., ge=0, description="Top.")
    w: int = Field(..., gt=0, description="Width.")
    h: int = Field(..., gt=0, description="Height.")


class Tracklet(BaseModel):
    """Stable track for one object (e.g. car) across frames in a segment."""

    track_id: str = Field(..., description="Stable ID, e.g. 'T07', 'Car_T07'.")
    bboxes: list[BBoxFrame] = Field(default_factory=list, description="Bboxes per frame.")
    label: str | None = Field(default=None, description="Optional class label.")


# --- Segment (internal: frames in memory; dataclass to hold BGR arrays) ---


@dataclass
class Segment:
    """
    A fixed-window segment of video with sampled frames. Frames are BGR numpy arrays
    (or None on read failure); not serialized.
    """

    video_id: str
    t_start: float
    t_end: float
    frame_times: list[float]
    frames: list[Any]  # list[ndarray | None], BGR; typing.Any for np.ndarray


# --- VideoChunk (output for JSONL / Qdrant) ---


class VideoChunk(BaseModel):
    """
    Timestamped chunk for embedding and indexing. No raw frames; index_text + fields.
    cv_meta: serialized CV/track JSON string for metadata storage (doc_meta).
    """

    video_id: str = Field(..., description="Video identifier.")
    t_start: float = Field(..., ge=0, description="Chunk start (s).")
    t_end: float = Field(..., ge=0, description="Chunk end (s).")
    index_text: str = Field(..., description="Text used for embeddings.")
    fields: dict[str, Any] = Field(default_factory=dict, description="Structured VLM output.")
    frame_times: list[float] = Field(default_factory=list, description="Sampled timestamps (s).")
    tracks: list[Tracklet] = Field(default_factory=list, description="Optional track data.")
    cv_meta: str = Field(default="[]", description="Serialized CV JSON (bboxes, labels, tracking IDs). Empty [] when no tracks.")
    hash: str = Field(..., description="Cache key: sha256(video_id, t_start, t_end, sampling, model, prompt).")

    model_config = {"arbitrary_types_allowed": True}


# --- Hashing and chunk assembly ---

SAMPLING_POLICY_PREFIX = "biased_edges"


def build_chunk_hash(
    video_id: str,
    t_start: float,
    t_end: float,
    frame_times: list[float],
    model_id: str,
    prompt_id: str,
    *,
    sampling_prefix: str = SAMPLING_POLICY_PREFIX,
) -> str:
    """
    Deterministic hash for caching. Inputs: video_id, time range, sampling (via frame_times count + policy),
    model_id, prompt_id.
    """
    # Encode sampling: policy + number of frames (order is implied by frame_times if needed)
    payload = (
        video_id
        + "|"
        + f"{t_start:.6f}|{t_end:.6f}"
        + "|"
        + sampling_prefix
        + "|"
        + str(len(frame_times))
        + "|"
        + model_id
        + "|"
        + prompt_id
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compose_index_text(
    caption: str,
    events: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    transcript_slice: str | None = None,
    ocr_text: str | None = None,
) -> str:
    """
    Compose index_text from caption + events + entities + optional transcript and OCR.
    """
    parts: list[str] = []
    if caption:
        parts.append(caption)
    for e in events:
        # Flatten event dict to a short line
        parts.append(json.dumps(e, ensure_ascii=False))
    for ent in entities:
        parts.append(json.dumps(ent, ensure_ascii=False))
    if transcript_slice:
        parts.append(transcript_slice)
    if ocr_text:
        parts.append(ocr_text)
    return "\n".join(parts).strip() or "(no content)"

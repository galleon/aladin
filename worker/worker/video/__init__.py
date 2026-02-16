"""
Video ingestion: MP4 → time-chunked documents for RAG (embedding + Qdrant).
"""
from .schemas import (
    BBoxFrame,
    Segment,
    Tracklet,
    VideoChunk,
    VLMResponse,
    build_chunk_hash,
    compose_index_text,
)
from .video_processor import segment_video
from .vlm_backend import DummyVLMBackend, VLMBackend
from .tracker import NoopTracker, Tracker, YOLOTracker, YOLOAPITracker
from .video_pipeline import run_video_pipeline

__all__ = [
    "BBoxFrame",
    "Segment",
    "Tracklet",
    "VideoChunk",
    "VLMResponse",
    "build_chunk_hash",
    "compose_index_text",
    "segment_video",
    "DummyVLMBackend",
    "VLMBackend",
    "NoopTracker",
    "Tracker",
    "YOLOTracker",
    "YOLOAPITracker",
    "run_video_pipeline",
]

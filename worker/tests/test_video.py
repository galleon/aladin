"""
Unit tests for video ingestion: frame timestamp sampling, chunk assembly, schema validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure worker (and shared) are on path when running from worker/ or repo root
_worker_root = Path(__file__).resolve().parent.parent
if str(_worker_root) not in sys.path:
    sys.path.insert(0, str(_worker_root))

from worker.video.schemas import (
    BBoxFrame,
    Tracklet,
    VideoChunk,
    VLMResponse,
    build_chunk_hash,
    compose_index_text,
)
from worker.video.video_processor import biased_edges_timestamps


# --- biased_edges_timestamps ---


def test_biased_edges_timestamps_basic():
    ts = biased_edges_timestamps(0.0, 4.0, 6)
    assert len(ts) == 6
    assert all(0.0 <= t <= 4.0 for t in ts)
    assert ts == sorted(ts)


def test_biased_edges_timestamps_deterministic():
    a = biased_edges_timestamps(1.0, 5.0, 10)
    b = biased_edges_timestamps(1.0, 5.0, 10)
    assert a == b


def test_biased_edges_timestamps_edges_inside_range():
    ts = biased_edges_timestamps(2.0, 10.0, 8)
    assert ts[0] >= 2.0
    assert ts[-1] <= 10.0


def test_biased_edges_timestamps_zero_span():
    ts = biased_edges_timestamps(3.0, 3.0, 5)
    assert len(ts) == 0


def test_biased_edges_timestamps_reversed_span():
    ts = biased_edges_timestamps(5.0, 2.0, 4)  # t_end < t_start
    assert len(ts) == 0


def test_biased_edges_timestamps_n_one():
    ts = biased_edges_timestamps(0.0, 1.0, 1)
    assert len(ts) == 1
    assert 0.0 <= ts[0] <= 1.0


# --- build_chunk_hash ---


def test_build_chunk_hash_deterministic():
    h1 = build_chunk_hash("v1", 0.0, 4.0, [0.5, 1.0, 2.0], "m1", "p1")
    h2 = build_chunk_hash("v1", 0.0, 4.0, [0.5, 1.0, 2.0], "m1", "p1")
    assert h1 == h2


def test_build_chunk_hash_changes_with_inputs():
    base = build_chunk_hash("v1", 0.0, 4.0, [0.5, 1.0], "m1", "p1")
    assert build_chunk_hash("v2", 0.0, 4.0, [0.5, 1.0], "m1", "p1") != base
    assert build_chunk_hash("v1", 1.0, 4.0, [0.5, 1.0], "m1", "p1") != base
    assert build_chunk_hash("v1", 0.0, 4.0, [0.5, 1.0, 2.0], "m1", "p1") != base
    assert build_chunk_hash("v1", 0.0, 4.0, [0.5, 1.0], "m2", "p1") != base
    assert build_chunk_hash("v1", 0.0, 4.0, [0.5, 1.0], "m1", "p2") != base


# --- compose_index_text ---


def test_compose_index_text_caption_only():
    t = compose_index_text("Hello.", [], [], None)
    assert "Hello." in t


def test_compose_index_text_with_events_and_entities():
    t = compose_index_text(
        "Scene.",
        [{"type": "step", "action": "turn"}],
        [{"id": "T01", "label": "car"}],
        None,
    )
    assert "Scene." in t
    assert "turn" in t
    assert "T01" in t


def test_compose_index_text_empty():
    t = compose_index_text("", [], [], None)
    assert t == "(no content)"


def test_compose_index_text_transcript_stub():
    t = compose_index_text("Cap", [], [], "Transcript slice.")
    assert "Transcript slice." in t


# --- Schema validation ---


def test_bbox_frame():
    b = BBoxFrame(t=1.5, x=10, y=20, w=100, h=50)
    assert b.t == 1.5
    assert b.model_dump()["x"] == 10


def test_tracklet_roundtrip():
    t = Tracklet(
        track_id="T07", bboxes=[BBoxFrame(t=0.0, x=0, y=0, w=50, h=50)], label="car"
    )
    d = t.model_dump()
    t2 = Tracklet.model_validate(d)
    assert t2.track_id == t.track_id
    assert len(t2.bboxes) == 1


def test_video_chunk_roundtrip():
    c = VideoChunk(
        video_id="v1",
        t_start=0.0,
        t_end=4.0,
        index_text="Summary.",
        fields={"caption": "x", "events": [], "model_id": "m", "prompt_id": "p"},
        frame_times=[0.5, 1.0],
        tracks=[],
        hash="abc123",
    )
    d = c.model_dump()
    c2 = VideoChunk.model_validate(d)
    assert c2.video_id == c.video_id
    assert c2.index_text == c.index_text


def test_video_chunk_jsonl_serializable():
    c = VideoChunk(
        video_id="v1",
        t_start=0.0,
        t_end=4.0,
        index_text="X",
        fields={"caption": "x", "model_id": "m", "prompt_id": "p"},
        frame_times=[],
        tracks=[],
        hash="h",
    )
    line = c.model_dump_json()
    parsed = json.loads(line)
    assert parsed["video_id"] == "v1"
    assert parsed["hash"] == "h"


def test_vlm_response():
    r = VLMResponse(
        caption="C", events=[], entities=[], notes=["n"], model_id="m", prompt_id="p"
    )
    d = r.model_dump()
    assert d["caption"] == "C"
    assert d["notes"] == ["n"]

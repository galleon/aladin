"""
Unit tests for worker.worker.video.ocr_processor pure functions.
No PaddleOCR installation required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_worker_root = Path(__file__).resolve().parent.parent
if str(_worker_root) not in sys.path:
    sys.path.insert(0, str(_worker_root))

from worker.video.ocr_processor import (
    OcrHit,
    box_to_aabb,
    iou_aabb,
    merge_hits,
    normalize_text,
    resize_max_side,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Return a 4-corner box as (4,2) float32 array."""
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )


def _make_hit(text: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> OcrHit:
    return OcrHit(text=text, conf=conf, box=_make_box(x1, y1, x2, y2), scale_tag="test")


# ---------------------------------------------------------------------------
# resize_max_side
# ---------------------------------------------------------------------------

class TestResizeMaxSide:
    def test_downscale_landscape(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized, factor = resize_max_side(img, 960)
        assert factor == pytest.approx(960 / 1920)
        assert resized.shape[1] == 960  # width is max side

    def test_downscale_portrait(self):
        img = np.zeros((3000, 1000, 3), dtype=np.uint8)
        resized, factor = resize_max_side(img, 2200)
        assert factor == pytest.approx(2200 / 3000)
        assert resized.shape[0] == 2200  # height is max side

    def test_no_op_when_smaller(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized, factor = resize_max_side(img, 1920)
        assert factor == 1.0
        assert resized is img  # same object, no copy

    def test_exact_max_side(self):
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        resized, factor = resize_max_side(img, 1000)
        assert factor == 1.0

    def test_output_dtype_preserved(self):
        img = np.zeros((2000, 3000, 3), dtype=np.uint8)
        resized, _ = resize_max_side(img, 1500)
        assert resized.dtype == np.uint8


# ---------------------------------------------------------------------------
# iou_aabb
# ---------------------------------------------------------------------------

class TestIouAabb:
    def test_no_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 30.0, 30.0)
        assert iou_aabb(a, b) == 0.0

    def test_full_overlap_identical(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert iou_aabb(box, box) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # Two 10×10 boxes overlapping by 5×10 = 50
        a = (0.0, 0.0, 10.0, 10.0)   # area 100
        b = (5.0, 0.0, 15.0, 10.0)   # area 100, intersection = 50
        iou = iou_aabb(a, b)
        assert iou == pytest.approx(50.0 / 150.0)

    def test_contained_box(self):
        outer = (0.0, 0.0, 10.0, 10.0)  # area 100
        inner = (2.0, 2.0, 8.0, 8.0)    # area 36, fully inside
        iou = iou_aabb(outer, inner)
        assert iou == pytest.approx(36.0 / 100.0)

    def test_touching_edges(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 0.0, 20.0, 10.0)  # shares edge at x=10
        assert iou_aabb(a, b) == 0.0


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        assert normalize_text("hello   world") == "hello world"

    def test_newlines_collapsed(self):
        assert normalize_text("foo\nbar") == "foo bar"

    def test_empty_string(self):
        assert normalize_text("   ") == ""

    def test_already_clean(self):
        assert normalize_text("clean text") == "clean text"


# ---------------------------------------------------------------------------
# merge_hits
# ---------------------------------------------------------------------------

class TestMergeHits:
    def test_empty(self):
        assert merge_hits([], merge_iou=0.35) == []

    def test_single_hit_kept(self):
        hits = [_make_hit("A", 0.9, 0, 0, 10, 10)]
        result = merge_hits(hits, merge_iou=0.35)
        assert len(result) == 1
        assert result[0].text == "A"

    def test_identical_boxes_deduplicated(self):
        hits = [
            _make_hit("hello", 0.9, 0, 0, 100, 20),
            _make_hit("hello", 0.8, 0, 0, 100, 20),  # same box, lower conf
        ]
        result = merge_hits(hits, merge_iou=0.35)
        assert len(result) == 1
        assert result[0].conf == pytest.approx(0.9)  # highest conf kept

    def test_non_overlapping_both_kept(self):
        hits = [
            _make_hit("top", 0.9, 0, 0, 100, 20),
            _make_hit("bottom", 0.85, 0, 200, 100, 220),
        ]
        result = merge_hits(hits, merge_iou=0.35)
        assert len(result) == 2

    def test_same_text_soft_iou_suppressed(self):
        # merge_iou=0.35 → soft threshold = 0.21
        # Boxes overlap ~25% → iou ≈ 0.25 > 0.21 and same text → suppress
        hits = [
            _make_hit("word", 0.9, 0, 0, 100, 20),
            _make_hit("word", 0.7, 75, 0, 175, 20),  # ~25% overlap
        ]
        result = merge_hits(hits, merge_iou=0.35)
        assert len(result) == 1

    def test_different_text_not_suppressed_by_soft_threshold(self):
        # Same moderate overlap but different text — only hard IoU applies
        hits = [
            _make_hit("foo", 0.9, 0, 0, 100, 20),
            _make_hit("bar", 0.7, 75, 0, 175, 20),  # overlap ~16% < 0.35 → kept
        ]
        result = merge_hits(hits, merge_iou=0.35)
        assert len(result) == 2

    def test_higher_conf_always_wins(self):
        hits = [
            _make_hit("low", 0.5, 0, 0, 100, 20),
            _make_hit("high", 0.95, 0, 0, 100, 20),  # same box, processed second
        ]
        # merge_hits sorts by conf descending before processing
        result = merge_hits(hits, merge_iou=0.35)
        assert len(result) == 1
        assert result[0].text == "high"


# ---------------------------------------------------------------------------
# box_to_aabb
# ---------------------------------------------------------------------------

class TestBoxToAabb:
    def test_axis_aligned(self):
        box = _make_box(5, 10, 50, 80)
        x1, y1, x2, y2 = box_to_aabb(box)
        assert x1 == pytest.approx(5)
        assert y1 == pytest.approx(10)
        assert x2 == pytest.approx(50)
        assert y2 == pytest.approx(80)

    def test_rotated_box(self):
        # Diamond-ish rotated box
        box = np.array([[10, 0], [20, 10], [10, 20], [0, 10]], dtype=np.float32)
        x1, y1, x2, y2 = box_to_aabb(box)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 20
        assert y2 == 20

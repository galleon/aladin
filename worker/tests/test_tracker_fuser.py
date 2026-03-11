"""
Tests for TrackletFuser: cross-segment track ID fusion via bbox IoU matching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_worker_root = Path(__file__).resolve().parent.parent
if str(_worker_root) not in sys.path:
    sys.path.insert(0, str(_worker_root))

from worker.video.schemas import BBoxFrame, Tracklet
from worker.video.tracker import TrackletFuser


def _make_tracklet(track_id: str, bboxes: list[BBoxFrame], label: str = "obj") -> Tracklet:
    return Tracklet(track_id=track_id, bboxes=bboxes, label=label)


def _bbox(t: float, x: int = 10, y: int = 10, w: int = 50, h: int = 50) -> BBoxFrame:
    return BBoxFrame(t=t, x=x, y=y, w=w, h=h)


# --- Tests ---


def test_single_segment_assigns_global_ids():
    """Single segment: all tracklets get G001, G002, ... assigned."""
    fuser = TrackletFuser()
    seg0 = [
        _make_tracklet("T01", [_bbox(0.0), _bbox(1.0)]),
        _make_tracklet("T02", [_bbox(0.5), _bbox(1.5)]),
    ]
    result = fuser.fuse([seg0])
    assert len(result) == 1
    gids = [t.global_track_id for t in result[0]]
    assert gids[0] == "G001"
    assert gids[1] == "G002"
    # local track_ids are preserved
    assert result[0][0].track_id == "T01"
    assert result[0][1].track_id == "T02"


def test_two_segments_overlapping_tracklet_same_global_id():
    """
    Two adjacent segments where one tracklet clearly overlaps (high IoU).
    The overlapping tracklet should receive the same global_track_id.
    """
    fuser = TrackletFuser(iou_threshold=0.4)

    # Segment 0: tracklet T01 ends near t=3.0 at bbox (10,10,50,50)
    seg0 = [_make_tracklet("T01", [_bbox(1.0), _bbox(3.0, x=10, y=10, w=50, h=50)])]

    # Segment 1: tracklet T01 starts near t=3.5 at nearly the same location -> high IoU
    seg1 = [_make_tracklet("T01", [_bbox(3.5, x=12, y=12, w=50, h=50), _bbox(5.0)])]

    result = fuser.fuse([seg0, seg1])

    gid_seg0 = result[0][0].global_track_id
    gid_seg1 = result[1][0].global_track_id

    assert gid_seg0 is not None
    assert gid_seg1 is not None
    assert gid_seg0 == gid_seg1, (
        f"Expected matching global IDs for overlapping tracklet, got {gid_seg0!r} vs {gid_seg1!r}"
    )


def test_two_segments_no_overlap_different_global_ids():
    """
    Two segments where tracklets are spatially far apart → IoU below threshold → new global IDs.
    """
    fuser = TrackletFuser(iou_threshold=0.4)

    # Segment 0: tracklet ends at top-left corner
    seg0 = [_make_tracklet("T01", [_bbox(1.0, x=0, y=0, w=30, h=30)])]

    # Segment 1: tracklet starts at bottom-right corner — no overlap
    seg1 = [_make_tracklet("T02", [_bbox(4.0, x=500, y=500, w=30, h=30)])]

    result = fuser.fuse([seg0, seg1])

    gid_seg0 = result[0][0].global_track_id
    gid_seg1 = result[1][0].global_track_id

    assert gid_seg0 is not None
    assert gid_seg1 is not None
    assert gid_seg0 != gid_seg1, (
        f"Expected different global IDs for non-overlapping tracklets, both got {gid_seg0!r}"
    )


def test_empty_segments_no_crash():
    """Empty segment list returns empty list without error."""
    fuser = TrackletFuser()
    result = fuser.fuse([])
    assert result == []


def test_segments_with_empty_track_lists():
    """Segments with no tracklets: no crash, returns empty inner lists."""
    fuser = TrackletFuser()
    result = fuser.fuse([[], []])
    assert result == [[], []]


def test_global_ids_format():
    """Global IDs must follow 'G{NNN}' zero-padded format."""
    fuser = TrackletFuser()
    seg0 = [_make_tracklet(f"T{i:02d}", [_bbox(float(i))]) for i in range(1, 12)]
    result = fuser.fuse([seg0])
    gids = [t.global_track_id for t in result[0]]
    assert gids[0] == "G001"
    assert gids[9] == "G010"
    assert gids[10] == "G011"


def test_multiple_segments_chain_same_object():
    """
    Three segments where the same object appears in each. Should carry the same
    global_track_id across all three.
    """
    fuser = TrackletFuser(iou_threshold=0.3)

    # Same object position drifts slightly between segments
    seg0 = [_make_tracklet("A", [_bbox(0.0, x=100, y=100, w=60, h=60), _bbox(3.0, x=105, y=105, w=60, h=60)])]
    seg1 = [_make_tracklet("B", [_bbox(3.5, x=108, y=108, w=60, h=60), _bbox(7.0, x=112, y=112, w=60, h=60)])]
    seg2 = [_make_tracklet("C", [_bbox(7.5, x=115, y=115, w=60, h=60), _bbox(11.0, x=120, y=120, w=60, h=60)])]

    result = fuser.fuse([seg0, seg1, seg2])

    gid0 = result[0][0].global_track_id
    gid1 = result[1][0].global_track_id
    gid2 = result[2][0].global_track_id

    assert gid0 == gid1 == gid2, (
        f"Expected same global ID across 3 segments, got {gid0!r}, {gid1!r}, {gid2!r}"
    )


def test_tracklet_bboxes_preserved():
    """Fuser must not drop or modify bbox data on the returned Tracklets."""
    fuser = TrackletFuser()
    original_bboxes = [_bbox(0.5, x=10, y=20, w=40, h=80), _bbox(1.0, x=15, y=25, w=40, h=80)]
    seg0 = [_make_tracklet("T01", original_bboxes)]
    result = fuser.fuse([seg0])
    returned = result[0][0]
    assert len(returned.bboxes) == 2
    assert returned.bboxes[0].x == 10
    assert returned.bboxes[1].x == 15


def test_one_to_one_matching_not_shared():
    """
    Two curr tracklets that both overlap the same prev tracklet should NOT share its global ID.
    The best-IoU match claims it; the second gets a fresh global ID.
    """
    fuser = TrackletFuser(iou_threshold=0.3)

    # Seg 0: one tracklet ending at a central position
    seg0 = [_make_tracklet("P1", [_bbox(3.0, x=100, y=100, w=60, h=60)])]

    # Seg 1: two tracklets both starting at nearly the same position as P1
    seg1 = [
        _make_tracklet("C1", [_bbox(3.5, x=102, y=102, w=60, h=60)]),  # high IoU with P1
        _make_tracklet("C2", [_bbox(3.5, x=105, y=105, w=60, h=60)]),  # also high IoU with P1
    ]

    result = fuser.fuse([seg0, seg1])
    gid_p1 = result[0][0].global_track_id
    gid_c1 = result[1][0].global_track_id
    gid_c2 = result[1][1].global_track_id

    # Exactly one curr tracklet should inherit P1's global ID; the other gets a new one
    assert gid_c1 != gid_c2, "Two curr tracklets must not share the same global ID"
    matched = sum(1 for g in (gid_c1, gid_c2) if g == gid_p1)
    assert matched == 1, f"Exactly one curr tracklet should inherit {gid_p1!r}, got {matched}"


def test_no_spatial_overlap_does_not_fuse():
    """
    High temporal proximity but zero spatial overlap must NOT produce a match,
    regardless of how close in time the segments are.
    """
    fuser = TrackletFuser(iou_threshold=0.4)

    # Seg 0 ends at top-left; Seg 1 starts at bottom-right — zero IoU
    seg0 = [_make_tracklet("A", [_bbox(2.9, x=0, y=0, w=40, h=40)])]
    seg1 = [_make_tracklet("B", [_bbox(3.0, x=960, y=540, w=40, h=40)])]

    result = fuser.fuse([seg0, seg1])
    assert result[0][0].global_track_id != result[1][0].global_track_id

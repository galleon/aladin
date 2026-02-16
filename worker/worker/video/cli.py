"""
CLI for video ingestion: python -m worker video <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure shared and sibling modules are importable when run from worker/
_here = Path(__file__).resolve().parent
_worker_root = _here.parent.parent
if str(_worker_root) not in sys.path:
    sys.path.insert(0, str(_worker_root))

from worker.video.video_pipeline import run_video_pipeline
from worker.video.vlm_backend import DummyVLMBackend


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest MP4 into time-chunked JSONL for RAG (embedding + Qdrant).",
    )
    parser.add_argument("video", type=Path, help="Path to MP4 file.")
    parser.add_argument(
        "--mode",
        choices=["procedure", "race"],
        default="procedure",
        help="VLM mode: procedure (steps) or race (per-track commentary).",
    )
    parser.add_argument(
        "--segment-sec", type=float, default=4.0, help="Segment window in seconds."
    )
    parser.add_argument(
        "--overlap-sec", type=float, default=1.0, help="Overlap between segments."
    )
    parser.add_argument(
        "--frames", type=int, default=10, help="Frames per segment (8–12)."
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--tracker",
        choices=["noop", "simple", "yolo", "yolo_api"],
        default="yolo",
        help="Tracker: yolo (local), yolo_api (Roboflow), simple (color-blob), or noop.",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1

    object_tracker = (
        "none" if args.tracker == "noop"
        else ("simple_blob" if args.tracker == "simple"
              else ("yolo_api" if args.tracker == "yolo_api" else "yolo"))
    )
    summary = run_video_pipeline(
        args.video,
        args.out,
        mode=args.mode,
        segment_sec=args.segment_sec,
        overlap_sec=args.overlap_sec,
        num_frames=args.frames,
        vlm=DummyVLMBackend(),
        object_tracker=object_tracker,
        enable_cv=(args.tracker != "noop"),
    )

    print(f"Segments: {summary['num_segments']}")
    print(f"Avg frames/segment: {summary['avg_frames_per_segment']}")
    print(f"Elapsed (s): {summary['elapsed_sec']}")
    print(f"Output: {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

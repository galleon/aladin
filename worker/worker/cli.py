"""
Unified CLI for ingestion tasks: python -m worker file|video|web [args...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure worker root is on path when run as main
_here = Path(__file__).resolve().parent
_worker_root = _here.parent
if str(_worker_root) not in sys.path:
    sys.path.insert(0, str(_worker_root))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingestion CLI: extract and chunk documents (file, video, web) to JSONL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Task type")

    # file
    file_parser = subparsers.add_parser("file", help="Extract and chunk a document (PDF, DOCX, TXT, etc.)")
    file_parser.add_argument("file", type=Path, help="Path to document.")
    file_parser.add_argument("--out", "-o", type=Path, default=Path("chunks.jsonl"), help="Output JSONL path.")
    file_parser.add_argument("--job-id", default="cli", help="Job ID for chunk IDs.")

    # video
    video_parser = subparsers.add_parser("video", help="Ingest MP4 into time-chunked JSONL")
    video_parser.add_argument("video", type=Path, help="Path to MP4 file.")
    video_parser.add_argument("--mode", choices=["procedure", "race"], default="procedure", help="VLM mode.")
    video_parser.add_argument("--segment-sec", type=float, default=4.0, help="Segment window (s).")
    video_parser.add_argument("--overlap-sec", type=float, default=1.0, help="Overlap between segments (s).")
    video_parser.add_argument("--frames", type=int, default=10, help="Frames per segment.")
    video_parser.add_argument("--out", "-o", type=Path, default=Path("chunks.jsonl"), help="Output JSONL path.")
    video_parser.add_argument(
        "--tracker",
        choices=["noop", "simple", "yolo", "yolo_api"],
        default="yolo",
        help="Tracker: noop, simple, yolo, or yolo_api.",
    )

    # web
    web_parser = subparsers.add_parser("web", help="Crawl a URL and chunk to JSONL")
    web_parser.add_argument("url", help="URL to crawl.")
    web_parser.add_argument("--out", "-o", type=Path, default=Path("chunks.jsonl"), help="Output JSONL path.")
    web_parser.add_argument("--depth", type=int, default=1, help="Crawl depth limit.")
    web_parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages to crawl.")
    web_parser.add_argument("--job-id", default="cli", help="Job ID for chunk IDs.")

    args = parser.parse_args()

    if args.command == "file":
        return _run_file(args)
    if args.command == "video":
        return _run_video(args)
    if args.command == "web":
        return _run_web(args)
    return 1


def _run_file(args) -> int:
    import asyncio

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1

    class _NoOpJobContext:
        async def update_job_status(self, *a, **kw):
            pass

    async def run():
        from worker.file.processor import FileProcessor

        processor = FileProcessor(
            job_ctx=_NoOpJobContext(),
            job_id=args.job_id,
            collection_name="cli",
            processing_config={},
        )
        chunks = await processor.extract_and_chunk(str(args.file), args.file.name, {})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
        return len(chunks)

    n = asyncio.run(run())
    print(f"Chunks: {n}")
    print(f"Output: {args.out}")
    return 0


def _run_video(args) -> int:
    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1

    object_tracker = (
        "none" if args.tracker == "noop"
        else ("simple_blob" if args.tracker == "simple"
              else ("yolo_api" if args.tracker == "yolo_api" else "yolo"))
    )
    from worker.video.video_pipeline import run_video_pipeline
    from worker.video.vlm_backend import DummyVLMBackend

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


def _run_web(args) -> int:
    import asyncio

    class _NoOpJobContext:
        async def update_job_status(self, *a, **kw):
            pass

    async def run():
        from worker.web.processor import WebProcessor

        processor = WebProcessor(
            job_ctx=_NoOpJobContext(),
            job_id=args.job_id,
            collection_name="cli",
        )
        source_config = {"depth_limit": args.depth, "max_pages": args.max_pages}
        chunks = await processor.crawl_and_chunk(args.url, source_config, {}, {})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
        return len(chunks)

    n = asyncio.run(run())
    print(f"Chunks: {n}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

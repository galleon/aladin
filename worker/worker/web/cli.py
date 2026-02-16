"""
CLI for web ingestion: python -m worker web <url>
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


class _NoOpJobContext:
    async def update_job_status(self, *args, **kwargs):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crawl a URL and chunk to JSONL (no embedding).",
    )
    parser.add_argument("url", help="URL to crawl.")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Crawl depth limit.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pages to crawl.",
    )
    parser.add_argument(
        "--job-id",
        default="cli",
        help="Job ID for chunk IDs.",
    )
    args = parser.parse_args()

    _here = Path(__file__).resolve().parent
    _worker_root = _here.parent.parent
    if str(_worker_root) not in sys.path:
        sys.path.insert(0, str(_worker_root))

    async def run():
        from worker.web.processor import WebProcessor

        job_ctx = _NoOpJobContext()
        processor = WebProcessor(
            job_ctx=job_ctx,
            job_id=args.job_id,
            collection_name="cli",
        )
        source_config = {
            "depth_limit": args.depth,
            "max_pages": args.max_pages,
        }
        chunks = await processor.crawl_and_chunk(
            args.url,
            source_config,
            {},
            {},
        )
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

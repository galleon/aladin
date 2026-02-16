"""
CLI for file ingestion: python -m worker file <path>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


class _NoOpJobContext:
    """Minimal job context for CLI (no Redis, no-op status updates)."""
    async def update_job_status(self, *args, **kwargs):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and chunk a document to JSONL (no embedding).",
    )
    parser.add_argument("file", type=Path, help="Path to document (PDF, DOCX, TXT, etc.).")
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--job-id",
        default="cli",
        help="Job ID for chunk IDs.",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1

    # Ensure worker root is on path
    _here = Path(__file__).resolve().parent
    _worker_root = _here.parent.parent
    if str(_worker_root) not in sys.path:
        sys.path.insert(0, str(_worker_root))

    async def run():
        from worker.file.processor import FileProcessor

        job_ctx = _NoOpJobContext()
        processor = FileProcessor(
            job_ctx=job_ctx,
            job_id=args.job_id,
            collection_name="cli",
            processing_config={},
        )
        chunks = await processor.extract_and_chunk(
            str(args.file),
            args.file.name,
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

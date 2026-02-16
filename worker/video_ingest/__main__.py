"""Run: python -m video_ingest <video.mp4> --mode procedure --out chunks.jsonl"""
from worker.video.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

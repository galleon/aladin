"""Run: python -m worker.video <video.mp4> --mode procedure --out chunks.jsonl"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

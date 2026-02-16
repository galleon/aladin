"""
Entry point for python -m worker.

  python -m worker file doc.pdf     -> CLI (extract + chunk)
  python -m worker video demo.mp4   -> CLI (video pipeline)
  python -m worker web https://...  -> CLI (crawl + chunk)
  python -m worker                 -> ARQ worker (job queue)
"""
import sys

if len(sys.argv) > 1:
    from worker.cli import main
    sys.exit(main())

# Run ARQ worker
from worker.main import WorkerSettings
from arq import run_worker
run_worker(WorkerSettings)

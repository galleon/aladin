#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""
Smoke-test every model endpoint used by aladin, across both DGX Spark nodes.

Usage:
    uv run scripts/check_services.py
    uv run scripts/check_services.py --spark1 192.168.1.72 --spark2 192.168.1.49
    uv run scripts/check_services.py --skip cosmos thinking   # skip slow models
"""

import argparse
import io
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import httpx

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

OK = f"{GREEN}✓ ok{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
SKIP = f"{YELLOW}– skip{RESET}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _silent_wav(duration_secs: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a silent 16-bit mono WAV."""
    num_samples = int(sample_rate * duration_secs)
    pcm = b"\x00\x00" * num_samples
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _probe(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs: Any,
) -> tuple[bool, str]:
    """Return (success, detail)."""
    try:
        r = client.request(method, url, **kwargs)
        if r.status_code < 400:
            return True, f"HTTP {r.status_code}"
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
    except httpx.TimeoutException:
        return False, "timeout"
    except httpx.ConnectError as e:
        return False, f"connection refused ({e})"
    except Exception as e:
        return False, str(e)


# ── Check definitions ─────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    node: str           # "local", "dgx1", "dgx2"
    key: str            # short skip key
    timeout: float = 30.0
    # filled by run()
    passed: bool = False
    detail: str = ""
    elapsed: float = 0.0
    skipped: bool = False


def run_checks(spark1: str, spark2: str, skip: set[str]) -> list[Check]:
    checks: list[Check] = []

    with httpx.Client(timeout=60.0) as client:

        def do(check: Check, method: str, url: str, **kwargs: Any) -> None:
            if check.key in skip:
                check.skipped = True
                checks.append(check)
                return
            c = httpx.Client(timeout=check.timeout)
            t0 = time.perf_counter()
            check.passed, check.detail = _probe(c, method, url, **kwargs)
            check.elapsed = time.perf_counter() - t0
            c.close()
            checks.append(check)

        # ── dgx1: spark-9965.local ────────────────────────────────────────────

        do(
            Check("granite-docling-258m (dgx1:8000)", "dgx1", "granite", timeout=60),
            "POST", f"http://{spark1}:8000/v1/chat/completions",
            json={
                "model": "granite-docling-258m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
        )

        do(
            Check("cosmos-reason2-8b / vlm (dgx1:8001)", "dgx1", "cosmos", timeout=60),
            "POST", f"http://{spark1}:8001/v1/chat/completions",
            json={
                "model": "cosmos-reason2-8b",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
        )

        do(
            Check("Riva ASR health (dgx1:9000)", "dgx1", "riva", timeout=10),
            "GET", f"http://{spark1}:9000/v1/health/ready",
        )

        # STT via dgx1 LiteLLM → stt-adapter → Riva gRPC
        wav = _silent_wav()
        do(
            Check("stt via dgx1 litellm (dgx1:4000)", "dgx1", "stt", timeout=30),
            "POST", f"http://{spark1}:4000/v1/audio/transcriptions",
            files={"file": ("silent.wav", wav, "audio/wav")},
            data={"model": "stt"},
        )

        # ── dgx2: spark-cedf.local ────────────────────────────────────────────

        do(
            Check("gpt-oss-20b / llm (dgx2:8002)", "dgx2", "llm", timeout=60),
            "POST", f"http://{spark2}:8002/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
        )

        do(
            Check("bge-large-en-v1.5 / embeddings (dgx2:8003)", "dgx2", "embeddings", timeout=30),
            "POST", f"http://{spark2}:8003/v1/embeddings",
            json={"model": "bge-large-en-v1.5", "input": "hello world"},
        )

        do(
            Check("bge-reranker-v2-m3 / reranker (dgx2:8004)", "dgx2", "reranker", timeout=30),
            "POST", f"http://{spark2}:8004/v1/score",
            json={
                "model": "bge-reranker-v2-m3",
                "text_1": "What is the capital of France?",
                "text_2": ["Paris is the capital of France."],
            },
        )

        do(
            Check("kokoro-82m / tts (dgx2:8011)", "dgx2", "tts", timeout=30),
            "POST", f"http://{spark2}:8011/v1/audio/speech",
            json={"model": "kokoro-82m", "input": "ok", "voice": "shimmer"},
        )

    return checks


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(checks: list[Check]) -> int:
    node_order = {"dgx1": 0, "dgx2": 1}
    checks_sorted = sorted(checks, key=lambda c: (node_order.get(c.node, 99), c.name))

    current_node = None
    for c in checks_sorted:
        if c.node != current_node:
            current_node = c.node
            label = {"dgx1": "dgx1 — spark-9965", "dgx2": "dgx2 — spark-cedf"}.get(c.node, c.node)
            print(f"\n{BOLD}{CYAN}{label}{RESET}")

        if c.skipped:
            status = SKIP
            timing = ""
        elif c.passed:
            status = OK
            timing = f"  {c.elapsed:.1f}s"
        else:
            status = FAIL
            timing = f"  {c.elapsed:.1f}s"

        detail = f"  {YELLOW}{c.detail}{RESET}" if not c.passed and not c.skipped else ""
        print(f"  {status}  {c.name}{timing}{detail}")

    failed = [c for c in checks if not c.passed and not c.skipped]
    print()
    if failed:
        print(f"{RED}{BOLD}{len(failed)} check(s) failed:{RESET}")
        for c in failed:
            print(f"  {RED}•{RESET} {c.name}: {c.detail}")
        return 1
    skipped = [c for c in checks if c.skipped]
    total = len(checks) - len(skipped)
    print(f"{GREEN}{BOLD}All {total} checks passed.{RESET}" + (f" ({len(skipped)} skipped)" if skipped else ""))
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test all aladin model endpoints")
    parser.add_argument("--spark1", default="spark-9965.local", help="dgx1 hostname or IP")
    parser.add_argument("--spark2", default="spark-cedf.local", help="dgx2 hostname or IP")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="KEY",
        help="Skip checks by key: granite cosmos riva stt llm embeddings reranker tts",
    )
    args = parser.parse_args()

    skip = set(args.skip or [])
    print(f"{BOLD}Checking aladin model endpoints…{RESET}")
    print(f"  dgx1 → {args.spark1}   dgx2 → {args.spark2}")
    if skip:
        print(f"  Skipping: {', '.join(sorted(skip))}")

    checks = run_checks(args.spark1, args.spark2, skip)
    sys.exit(print_results(checks))


if __name__ == "__main__":
    main()

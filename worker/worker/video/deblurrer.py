"""
Deblurrer protocol and implementations for optional frame preprocessing before VLM/CV.

API contract for DeblurGANInceptionDeblurrer (when using remote server):
  POST {DEBLUR_API_BASE}/deblur
  Content-Type: application/json
  Body: {"images": ["<base64_bgr>", ...]}
  Response: {"images": ["<base64_bgr>", ...]}  (same order, same count)
  Optional: Authorization: Bearer {DEBLUR_API_KEY}
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.request
from typing import Any, Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Deblurrer(Protocol):
    """Protocol for frame deblurring. Implementations process BGR numpy frames before VLM/CV."""

    def deblur_frames(self, frames_bgr: list[Any]) -> list[Any]:
        """Process a list of BGR numpy frames. Returns same-length list; None entries are passed through."""
        ...


class NoopDeblurrer:
    """No-op deblurrer: returns frames unchanged."""

    def deblur_frames(self, frames_bgr: list[Any]) -> list[Any]:
        return frames_bgr


class DeblurGANInceptionDeblurrer:
    """
    Deblur frames via remote DeblurGAN-Inception API.

    Expects DEBLUR_API_BASE (and optionally DEBLUR_API_KEY) in env.
    Falls back to NoopDeblurrer behavior when API is not configured or on failure.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        timeout: float = 60.0,
    ):
        try:
            from shared.config import settings
            self.api_base = (api_base or getattr(settings, "DEBLUR_API_BASE", None) or "").rstrip("/")
            self.api_key = api_key or getattr(settings, "DEBLUR_API_KEY", "") or ""
            self.model_id = model_id or getattr(settings, "DEBLUR_MODEL", "deblur-gan-inception") or "deblur-gan-inception"
        except Exception:
            self.api_base = (api_base or "").rstrip("/")
            self.api_key = api_key or ""
            self.model_id = model_id or "deblur-gan-inception"
        self.timeout = timeout

    def deblur_frames(self, frames_bgr: list[Any]) -> list[Any]:
        if not self.api_base:
            logger.debug("DEBLUR_API_BASE not configured; passing frames through")
            return frames_bgr

        # Collect valid (frame, index) pairs; preserve None at original indices
        valid: list[tuple[int, np.ndarray]] = []
        for i, f in enumerate(frames_bgr):
            if f is not None and isinstance(f, np.ndarray):
                valid.append((i, f))

        if not valid:
            return frames_bgr

        try:
            # Encode BGR frames to base64
            encoded = []
            for _, frame in valid:
                _, buf = cv2.imencode(".jpg", frame)
                encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))

            url = f"{self.api_base}/deblur"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {"images": encoded}
            if self.model_id:
                payload["model"] = self.model_id

            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            result_images = data.get("images", [])
            if len(result_images) != len(valid):
                logger.warning(
                    "Deblur API returned %d images, expected %d; passing original frames through",
                    len(result_images),
                    len(valid),
                )
                return frames_bgr

            # Reconstruct output list: fill in deblurred frames at original indices
            out: list[Any] = list(frames_bgr)  # copy, preserves None at each index
            for (orig_idx, _), b64 in zip(valid, result_images):
                try:
                    raw = base64.b64decode(b64)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        out[orig_idx] = decoded
                except Exception as e:
                    logger.debug("Failed to decode deblurred frame at index %d: %s", orig_idx, e)

            return out
        except Exception as e:
            logger.warning("Deblur API call failed: %s; passing original frames through", e)
            return frames_bgr


def _get_deblurrer(name: str) -> Deblurrer:
    """Factory: returns Deblurrer by name. 'inception' -> DeblurGANInceptionDeblurrer, else NoopDeblurrer."""
    if name == "inception":
        try:
            from shared.config import settings
            api_base = getattr(settings, "DEBLUR_API_BASE", None) or ""
            if not api_base:
                logger.warning("DEBLUR_API_BASE not set for inception deblurrer; using NoopDeblurrer")
                return NoopDeblurrer()
            return DeblurGANInceptionDeblurrer()
        except Exception as e:
            logger.warning("Failed to create DeblurGANInceptionDeblurrer: %s; using NoopDeblurrer", e)
            return NoopDeblurrer()
    return NoopDeblurrer()

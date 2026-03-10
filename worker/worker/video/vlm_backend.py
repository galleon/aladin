"""
VLMBackend protocol and implementations: Dummy, Cosmos (placeholder), OpenAICompatible.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Protocol

import cv2
import numpy as np
from openai import OpenAI

from .schemas import Tracklet
from .prompts import PROCEDURE_PROMPT_ID, RACE_PROMPT_ID, get_prompt, is_cosmos_reason2_model

logger = logging.getLogger(__name__)


class VLMBackend(Protocol):
    """
    Protocol for VLM analysis. Swap in Cosmos (external API) or on-prem backend.
    """

    def analyze_segment(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        frame_times: list[float],
        frames_bgr: list[Any],  # list[ndarray | None]
        mode: str,
        tracks: list[Tracklet] | None = None,
        extra_context: str | None = None,
        custom_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a segment. Returns dict with at least:
        - caption: str
        - events: list
        - entities: list
        - notes: list
        - model_id: str
        - prompt_id: str
        """
        ...


class DummyVLMBackend:
    """
    No-op backend with deterministic output for testing. No real API calls.
    """

    def __init__(self, model_id: str = "dummy-v1", prompt_id: str | None = None):
        self.model_id = model_id
        self._prompt_id = prompt_id

    def _prompt_id_for_mode(self, mode: str) -> str:
        if self._prompt_id:
            return self._prompt_id
        return PROCEDURE_PROMPT_ID if mode == "procedure" else RACE_PROMPT_ID

    def analyze_segment(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        frame_times: list[float],
        frames_bgr: list[Any],
        mode: str,
        tracks: list[Tracklet] | None = None,
        extra_context: str | None = None,
        custom_prompt: str | None = None,
    ) -> dict[str, Any]:
        n = len([f for f in frames_bgr if f is not None])
        prompt_id = self._prompt_id_for_mode(mode)
        cap = f"[{mode}] {video_id} [{t_start:.1f}s–{t_end:.1f}s] ({n} frames)."
        events = [{"type": "dummy", "t": t_start, "description": f"Segment at {t_start:.1f}s"}]
        entities = [{"id": "dummy_entity", "role": "placeholder"}]
        if tracks:
            for t in tracks:
                entities.append({"id": t.track_id, "label": t.label or "object"})
        return {
            "caption": cap,
            "events": events,
            "entities": entities,
            "notes": ["DummyVLMBackend: no real analysis."],
            "model_id": self.model_id,
            "prompt_id": prompt_id,
        }


# cosmos-reason2-8b: max context 32768 tokens. At 640px max side (640×360 for 16:9),
# each frame is ~230k pixels ≈ 1.2k visual tokens; 16 frames ≈ 19k — well within budget.
_COSMOS_R2_DEFAULT_MAX_SIDE = 640
_COSMOS_R2_DEFAULT_MAX_FRAMES = 16  # VSS uses 20; 16 is safe at 640px within 32k context
_COSMOS_R2_MAX_FRAMES_CAP = 20      # hard cap matching vLLM limit_mm_per_prompt guidance
_DEFAULT_MAX_FRAMES = 8             # non-cosmos default
_COSMOS_R2_SYSTEM_PROMPT = "You are a video analysis assistant. Output a structured JSON response."
# Reasoning format instruction appended to the user prompt (not system) for cosmos-reason2.
# Per NVIDIA VSS pattern: reasoning instruction in user prompt is more reliably followed by vLLM.
_COSMOS_R2_REASONING_SUFFIX = (
    "\n\nAnswer using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final JSON answer immediately after the </think> tag."
)


def _resize_frame(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Resize frame so its longest side is at most max_side, preserving aspect ratio."""
    h, w = frame_bgr.shape[:2]
    if max(h, w) <= max_side:
        return frame_bgr
    scale = max_side / max(h, w)
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _frame_to_base64_jpeg(frame_bgr: np.ndarray, max_side: int = 0) -> str:
    """Encode BGR numpy frame to base64 JPEG. Resizes to max_side if > 0."""
    if max_side > 0:
        frame_bgr = _resize_frame(frame_bgr, max_side)
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok or buf is None:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_json_object_from_text(text: str) -> str | None:
    """Extract a single top-level JSON object from text (handles ```json ... ``` or raw {...})."""
    text = text.strip()
    if not text:
        return None

    # Try ```json ... ``` or ``` ... ```: find content between first and last ```
    for marker in ("```json", "```"):
        idx = text.find(marker)
        if idx == -1:
            continue
        start = idx + len(marker)
        rest = text[start:].lstrip()
        # Find first { and then matching closing } by brace count
        brace = rest.find("{")
        if brace == -1:
            continue
        depth = 0
        for i, c in enumerate(rest[brace:]):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return rest[brace : brace + i + 1]
        # No matching close; try from first { to end
        try:
            return rest[brace:]
        except Exception:
            continue

    # Try raw first { ... } with brace matching
    brace = text.find("{")
    if brace == -1:
        return None
    depth = 0
    for i, c in enumerate(text[brace:], start=brace):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace : i + 1]
    return None


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _extract_think_block(text: str) -> tuple[str, str]:
    """
    Extract think block from VLM response (e.g. cosmos-reason2).
    Returns (reasoning_text, caption_text). Caption is the content after think block, used for embedding.
    Reasoning is stored for UI/debug.
    """
    if THINK_OPEN not in text or THINK_CLOSE not in text:
        return ("", text.strip())
    start = text.find(THINK_OPEN)
    end = text.find(THINK_CLOSE)
    if start == -1 or end == -1 or end <= start:
        return ("", text.strip())
    reasoning = text[start + len(THINK_OPEN) : end].strip()
    caption = text[end + len(THINK_CLOSE) :].strip()
    return (reasoning, caption)


def _salvage_caption_from_truncated(json_str: str) -> str:
    """Extract caption value from truncated JSON (e.g. finish_reason=length)."""
    m = re.search(r'"caption"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
    if m:
        return m.group(1).replace('\\"', '"') if m.group(1) else ""
    m = re.search(r'"caption"\s*:\s*"(.*)$', json_str, re.DOTALL)
    if m:
        return m.group(1).replace('\\"', '"').strip()
    return ""


def _parse_vlm_response(text: str) -> dict[str, Any]:
    """
    Parse VLM response text into a structured dict.
    Strips think block (reasoning) from cosmos-reason2-style output; caption used for embedding.
    """
    defaults: dict[str, Any] = {
        "caption": "",
        "events": [],
        "entities": [],
        "notes": [],
    }
    text = text.strip()
    if not text:
        return dict(defaults)

    reasoning, caption_part = _extract_think_block(text)
    text_to_parse = caption_part if caption_part else text

    json_str = _extract_json_object_from_text(text_to_parse)
    if not json_str:
        out = {**defaults, "caption": text_to_parse}
        if reasoning:
            out["reasoning"] = reasoning
        return out

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            return {**defaults, "caption": text}
        # Preserve all keys from the model (prompt-dependent); only include JSON-serializable values
        out = dict(defaults)
        for k, v in parsed.items():
            if isinstance(k, str) and isinstance(
                v, (str, list, dict, type(None), bool, int, float)
            ):
                out[k] = v
        # Ensure required keys exist for compose_index_text (inspect object, no hardcoding)
        if "caption" not in out and isinstance(out.get("summary"), str):
            out.setdefault("caption", out["summary"])
        out.setdefault("caption", "")
        if "events" not in out and isinstance(out.get("steps"), list):
            out.setdefault("events", out["steps"])
        out.setdefault("events", [])
        out.setdefault("entities", [])
        out.setdefault("notes", [])
        if reasoning:
            out["reasoning"] = reasoning
        return out
    except json.JSONDecodeError:
        pass

    # Truncated or malformed JSON: try to salvage caption
    caption = _salvage_caption_from_truncated(json_str)
    if caption:
        out = {**defaults, "caption": caption, "notes": ["Response truncated (max_tokens)."]}
        if reasoning:
            out["reasoning"] = reasoning
        return out
    out = {**defaults, "caption": text_to_parse}
    if reasoning:
        out["reasoning"] = reasoning
    return out


def unpack_vlm_for_log(vlm_out: dict[str, Any]) -> dict[str, Any]:
    """
    Unpack vlm_out for logging: if 'caption' contains a ```json ... ``` block,
    parse it and return the full dict (all keys from the model, no hardcoding).
    Preserves model_id, prompt_id. Use for human-readable log output.
    """
    result = dict(vlm_out)
    caption = result.get("caption", "")
    if not isinstance(caption, str) or "```" not in caption or "{" not in caption:
        return result
    json_str = _extract_json_object_from_text(caption)
    if not json_str:
        return result
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            result = {**parsed, "model_id": result.get("model_id", ""), "prompt_id": result.get("prompt_id", "")}
    except json.JSONDecodeError:
        pass
    return result


class OpenAICompatibleVLMBackend:
    """
    Vision API backend compatible with OpenAI chat/completions (e.g. GPT-4V, LLaVA).
    Encodes frames to base64 JPEG and sends to {api_base}/chat/completions.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str | None = None,
        model_id: str = "gpt-4o",
        max_side: int = 0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "sk-dummy"
        self.model_id = model_id
        # 0 = auto: uses _COSMOS_R2_DEFAULT_MAX_SIDE for cosmos-reason2, no resize otherwise
        self.max_side = max_side

    def analyze_segment(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        frame_times: list[float],
        frames_bgr: list[Any],
        mode: str,
        tracks: list[Tracklet] | None = None,
        extra_context: str | None = None,
        custom_prompt: str | None = None,
    ) -> dict[str, Any]:
        is_cosmos = is_cosmos_reason2_model(self.model_id)
        # Frame resize: auto-detect per model when max_side == 0.
        effective_max_side = self.max_side if self.max_side > 0 else (
            _COSMOS_R2_DEFAULT_MAX_SIDE if is_cosmos else 0
        )

        # Max frames: env var > auto-detect per model.
        try:
            from shared.config import settings as _settings
            _cfg_max = _settings.VLM_MAX_FRAMES
        except Exception:
            _cfg_max = 0
        if _cfg_max > 0:
            max_images = min(_cfg_max, _COSMOS_R2_MAX_FRAMES_CAP if is_cosmos else _cfg_max)
        else:
            max_images = _COSMOS_R2_DEFAULT_MAX_FRAMES if is_cosmos else _DEFAULT_MAX_FRAMES

        prompt_id = PROCEDURE_PROMPT_ID if mode == "procedure" else RACE_PROMPT_ID
        num_frames = len([f for f in frames_bgr if f is not None])
        tracks_json = extra_context
        if custom_prompt and custom_prompt.strip():
            prompt = (
                custom_prompt.strip()
                .replace("{t_start}", f"{t_start:.1f}")
                .replace("{t_end}", f"{t_end:.1f}")
                .replace("{num_frames}", str(num_frames))
                .replace("{tracks_json}", tracks_json or "")
            )
        else:
            prompt = get_prompt(mode, t_start, t_end, num_frames, tracks_json, model_id=self.model_id)

        # Encode frames (resized) — images go FIRST in content (NVIDIA ordering)
        valid_frames = [
            (f, t) for f, t in zip(frames_bgr, frame_times)
            if f is not None and isinstance(f, np.ndarray)
        ]
        image_content: list[dict[str, Any]] = []
        used_times: list[float] = []
        for frame, t in valid_frames[:max_images]:
            b64 = _frame_to_base64_jpeg(frame, max_side=effective_max_side)
            if b64:
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
                used_times.append(t)

        if not image_content:
            return {
                "caption": f"[No frames] {video_id} [{t_start:.1f}s–{t_end:.1f}s]",
                "events": [],
                "entities": [],
                "notes": ["No valid frames to analyze."],
                "model_id": self.model_id,
                "prompt_id": prompt_id,
            }

        # Prepend per-frame timestamps so the model can ground events to exact times
        ts_str = ", ".join(f"{t:.2f}s" for t in used_times)
        timestamp_header = (
            f"These {len(image_content)} frames were sampled from the video segment "
            f"[{t_start:.1f}s\u2013{t_end:.1f}s] at timestamps: {ts_str}."
        )
        full_prompt = f"{timestamp_header}\n\n{prompt}"

        # For cosmos-reason2: append reasoning format to user prompt (not system prompt).
        # Per NVIDIA VSS pattern, vLLM-served models follow user prompt instructions more reliably.
        if is_cosmos:
            full_prompt = full_prompt + _COSMOS_R2_REASONING_SUFFIX

        # Images before text (NVIDIA/cosmos-reason2 pattern)
        content: list[dict[str, Any]] = image_content + [{"type": "text", "text": full_prompt}]

        # Minimal system prompt for cosmos-reason2; reasoning instruction is in user prompt above.
        messages: list[dict[str, Any]] = []
        if is_cosmos:
            messages.append({"role": "system", "content": _COSMOS_R2_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": content})

        try:
            base = self.api_base.rstrip("/")
            base_url = base if base.endswith("v1") else f"{base}/v1"
            client = OpenAI(api_key=self.api_key, base_url=base_url)
            # no_repeat_ngram_size=3 reduces repetitive captions (vLLM sampling param, cosmos-reason2 only)
            extra_body = {"no_repeat_ngram_size": 3} if is_cosmos else {}
            resp = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=2048,
                extra_body=extra_body or None,
            )
            # Log raw API response (formatted for review)
            log_payload = {
                "id": getattr(resp, "id", None),
                "model": getattr(resp, "model", None),
                "usage": (
                    {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": getattr(resp.usage, "total_tokens", None),
                    }
                    if getattr(resp, "usage", None)
                    else None
                ),
                "choices": [
                    {
                        "index": c.index,
                        "message": {
                            "role": getattr(c.message, "role", None),
                            "content": getattr(c.message, "content", None),
                        },
                        "finish_reason": getattr(c, "finish_reason", None),
                    }
                    for c in (resp.choices or [])
                ],
            }
            logger.info(
                "VLM API response video_id=%s t_start=%.1f t_end=%.1f\n%s",
                video_id,
                t_start,
                t_end,
                json.dumps(log_payload, indent=2, ensure_ascii=False),
            )
            text = resp.choices[0].message.content if resp.choices else ""
            parsed = _parse_vlm_response(text)
            # Log extracted dict so it is easy to find (grep VLM_EXTRACTED_JSON)
            logger.info(
                "VLM_EXTRACTED_JSON video_id=%s t_start=%.1f t_end=%.1f",
                video_id,
                t_start,
                t_end,
            )
            logger.info("VLM_EXTRACTED_JSON dict:\n%s", json.dumps(parsed, indent=2, ensure_ascii=False))
            parsed["model_id"] = self.model_id
            parsed["prompt_id"] = prompt_id
            return parsed
        except Exception as e:
            logger.warning("VLM API call failed: %s", e)
            return {
                "caption": f"[VLM error] {video_id} [{t_start:.1f}s–{t_end:.1f}s]: {e!s}",
                "events": [],
                "entities": [],
                "notes": [f"VLM API failed: {e!s}"],
                "model_id": self.model_id,
                "prompt_id": prompt_id,
            }


class CosmosVLMBackend:
    """
    Placeholder for Cosmos external API. When vlm_api_base is set, use OpenAICompatibleVLMBackend instead.
    """

    def __init__(
        self,
        api_base: str = "https://api.cosmos.example/v1",
        api_key: str | None = None,
        model_id: str = "cosmos-default",
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model_id = model_id

    def analyze_segment(
        self,
        video_id: str,
        t_start: float,
        t_end: float,
        frame_times: list[float],
        frames_bgr: list[Any],
        mode: str,
        tracks: list[Tracklet] | None = None,
        extra_context: str | None = None,
        custom_prompt: str | None = None,
    ) -> dict[str, Any]:
        return {
            "caption": f"[Cosmos placeholder] {video_id} [{t_start:.1f}s–{t_end:.1f}s]",
            "events": [],
            "entities": [],
            "notes": ["CosmosVLMBackend: API not implemented. Use OpenAICompatibleVLMBackend."],
            "model_id": self.model_id,
            "prompt_id": PROCEDURE_PROMPT_ID if mode == "procedure" else RACE_PROMPT_ID,
        }

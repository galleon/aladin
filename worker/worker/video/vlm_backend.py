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
from .prompts import PROCEDURE_PROMPT_ID, RACE_PROMPT_ID, get_prompt

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


def _frame_to_base64_jpeg(frame_bgr: np.ndarray) -> str:
    """Encode BGR numpy frame to base64 JPEG."""
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
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or "sk-dummy"
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

        # Build content: text + images (up to 8 frames to avoid token limits)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        valid_frames = [(f, t) for f, t in zip(frames_bgr, frame_times) if f is not None and isinstance(f, np.ndarray)]
        max_images = 8
        for i, (frame, t) in enumerate(valid_frames[:max_images]):
            b64 = _frame_to_base64_jpeg(frame)
            if b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        if not content or (len(content) == 1 and not valid_frames):
            return {
                "caption": f"[No frames] {video_id} [{t_start:.1f}s–{t_end:.1f}s]",
                "events": [],
                "entities": [],
                "notes": ["No valid frames to analyze."],
                "model_id": self.model_id,
                "prompt_id": prompt_id,
            }

        try:
            base = self.api_base.rstrip("/")
            base_url = base if base.endswith("v1") else f"{base}/v1"
            client = OpenAI(api_key=self.api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
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

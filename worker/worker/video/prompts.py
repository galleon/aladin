"""
Prompt templates for VLM analysis. Loads from prompts.yaml when available.
"""
from __future__ import annotations

from ..prompts_loader import _get_prompts_data, get_video_prompt


def _prompt_ids() -> tuple[str, str]:
    d = _get_prompts_data()
    v = d.get("video", {})
    p = v.get("procedure", {})
    r = v.get("race", {})
    return (
        p.get("id", "procedure_v1") if isinstance(p, dict) else "procedure_v1",
        r.get("id", "race_v1") if isinstance(r, dict) else "race_v1",
    )


PROCEDURE_PROMPT_ID, RACE_PROMPT_ID = _prompt_ids()


def is_cosmos_reason2_model(model_id: str | None) -> bool:
    """Check if model_id indicates cosmos-reason2 (for timestamp instruction)."""
    if not model_id:
        return False
    return "cosmos-reason2" in model_id.lower()


def get_prompt(
    mode: str,
    t_start: float,
    t_end: float,
    num_frames: int,
    tracks_json: str | None,
    model_id: str | None = None,
) -> str:
    """Get VLM prompt for the given mode. Loads from YAML config."""
    prompt_text, _ = get_video_prompt(mode, t_start, t_end, num_frames, tracks_json, model_id)
    return prompt_text

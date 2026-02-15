"""
Load prompt templates from YAML config. Falls back to built-in defaults if file missing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Resolve path: worker/worker/prompts_loader.py -> worker/config/prompts.yaml
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_PROMPTS_PATH = _CONFIG_DIR / "prompts.yaml"

_BuiltinDefaults: dict[str, Any] = {
    "video": {
        "procedure": {
            "id": "procedure_v1",
            "template": (
                "Analyze this procedure video segment [{t_start:.1f}s–{t_end:.1f}s] "
                "using {num_frames} keyframes. For each step, provide:\n"
                "- **action**: what is being done\n"
                "- **tools**: tools/equipment used\n"
                "- **result**: outcome or visible state change\n"
                "- **warnings**: safety or quality cues if any\n"
                "- **visible_cues**: text, labels, or indicators on screen\n"
                'Respond with a structured JSON: { "caption": "...", "events": [...], '
                '"entities": [...], "notes": [...] }.'
            ),
            "tracks_append": "\nOptional tracking context (not required for procedure): {tracks_json}",
        },
        "race": {
            "id": "race_v1",
            "template": (
                "Analyze this race/ traffic video segment [{t_start:.1f}s–{t_end:.1f}s] "
                "using {num_frames} keyframes. Provide:\n"
                "- **caption**: scene-level summary\n"
                "- **events**: interactions (e.g. closing gap, overtake attempt, lane change), "
                "with per-track references when possible (e.g. 'Car T07', 'Pedestrian P01')\n"
                "- **entities**: vehicles, pedestrians, or other moving objects with stable IDs if known\n"
                "- **notes**: uncertainty when objects are occluded or leave frame\n"
                'Respond with a structured JSON: { "caption": "...", "events": [...], '
                '"entities": [...], "notes": [...] }.'
            ),
            "tracks_append": "\nTrack data (use IDs like T07 in commentary): {tracks_json}",
            "tracks_append_empty": "\nNo track data: use scene-level commentary only.",
        },
        "cosmos_reason2_timestamp_instruction": " Make sure the answer contain correct timestamps.",
    },
    "file": {
        "docling_convert": "Convert this page to docling.",
    },
}


def _load_yaml() -> dict[str, Any]:
    """Load prompts from YAML file or return built-in defaults."""
    if not _PROMPTS_PATH.exists():
        logger.debug("Prompts config not found at %s, using built-in defaults", _PROMPTS_PATH)
        return _BuiltinDefaults.copy()

    try:
        import yaml
        with open(_PROMPTS_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return _BuiltinDefaults.copy()
        # Deep merge with defaults so missing keys fall back
        return _deep_merge(_BuiltinDefaults.copy(), data)
    except Exception as e:
        logger.warning("Failed to load prompts from %s: %s. Using built-in defaults.", _PROMPTS_PATH, e)
        return _BuiltinDefaults.copy()


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Override values take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _get_prompts_data() -> dict[str, Any]:
    """Lazy load prompts data (cached)."""
    if not hasattr(_get_prompts_data, "_cache"):
        _get_prompts_data._cache = _load_yaml()
    return _get_prompts_data._cache


def get_prompt(key: str, **kwargs: Any) -> str:
    """
    Get a generic prompt by dotted key (e.g. 'file.docling_convert').
    Returns the template string or empty string if not found.
    """
    data = _get_prompts_data()
    parts = key.split(".")
    for part in parts:
        data = data.get(part) if isinstance(data, dict) else None
        if data is None:
            return ""
    if isinstance(data, str):
        return data.format(**kwargs) if kwargs else data
    return ""


def get_video_prompt(
    mode: str,
    t_start: float,
    t_end: float,
    num_frames: int,
    tracks_json: str | None,
    model_id: str | None = None,
) -> tuple[str, str]:
    """
    Get video prompt for the given mode. Returns (prompt_text, prompt_id).
    """
    data = _get_prompts_data()
    video = data.get("video", {})
    mode_config = video.get(mode, video.get("procedure", {}))

    if isinstance(mode_config, dict):
        template = mode_config.get("template", "")
        prompt_id = mode_config.get("id", "procedure_v1")
        tracks_append = mode_config.get("tracks_append", "")
        tracks_append_empty = mode_config.get("tracks_append_empty", "")
    else:
        template = ""
        prompt_id = "procedure_v1"
        tracks_append = ""
        tracks_append_empty = ""

    if not template:
        template = video.get("procedure", {}).get("template", "") if isinstance(video.get("procedure"), dict) else ""

    base = template.format(
        t_start=t_start,
        t_end=t_end,
        num_frames=num_frames,
        tracks_json=tracks_json or "",
    )

    if tracks_json and tracks_json.strip():
        base += tracks_append.format(tracks_json=tracks_json)
    elif tracks_append_empty:
        base += tracks_append_empty

    cosmos_instruction = video.get("cosmos_reason2_timestamp_instruction", "")
    if cosmos_instruction and model_id and "cosmos-reason2" in model_id.lower():
        base += cosmos_instruction

    return (base, prompt_id)

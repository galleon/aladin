"""Unit tests for VLM response parsing helpers in vlm_backend.py."""
import pytest

from worker.video.vlm_backend import (
    _extract_think_block,
    _parse_vlm_response,
    _strip_wrapper_tags,
)


# ---------------------------------------------------------------------------
# _strip_wrapper_tags
# ---------------------------------------------------------------------------

def test_strip_wrapper_tags_removes_answer():
    assert _strip_wrapper_tags("<answer>hello</answer>") == "hello"


def test_strip_wrapper_tags_removes_summary():
    assert _strip_wrapper_tags("<summary>text</summary>") == "text"


def test_strip_wrapper_tags_no_tags():
    assert _strip_wrapper_tags("plain text") == "plain text"


def test_strip_wrapper_tags_mixed():
    text = "<answer>some json</answer> extra"
    assert _strip_wrapper_tags(text) == "some json extra"


# ---------------------------------------------------------------------------
# _extract_think_block — Case 1: both tags present
# ---------------------------------------------------------------------------

def test_case1_both_tags():
    text = "<think>my reasoning</think>final answer"
    reasoning, caption = _extract_think_block(text)
    assert reasoning == "my reasoning"
    assert caption == "final answer"


def test_case1_both_tags_with_whitespace():
    text = "  <think>  reasoning text  </think>  caption text  "
    reasoning, caption = _extract_think_block(text)
    assert reasoning == "reasoning text"
    assert caption == "caption text"


def test_case1_strips_wrapper_tags_from_caption():
    text = "<think>reasoning</think><answer>real answer</answer>"
    reasoning, caption = _extract_think_block(text)
    assert reasoning == "reasoning"
    assert caption == "real answer"


def test_case1_multiline_reasoning():
    text = "<think>\nline1\nline2\n</think>\n{\"caption\": \"hello\"}"
    reasoning, caption = _extract_think_block(text)
    assert "line1" in reasoning
    assert "line2" in reasoning
    assert '{"caption": "hello"}' in caption


# ---------------------------------------------------------------------------
# _extract_think_block — Case 2: only </think> present
# ---------------------------------------------------------------------------

def test_case2_only_close_tag():
    text = "implicit reasoning here</think>final answer"
    reasoning, caption = _extract_think_block(text)
    assert reasoning == "implicit reasoning here"
    assert caption == "final answer"


def test_case2_only_close_tag_strips_wrappers():
    text = "reasoning</think><answer>clean answer</answer>"
    reasoning, caption = _extract_think_block(text)
    assert reasoning == "reasoning"
    assert caption == "clean answer"


def test_case2_empty_caption_after_close():
    text = "reasoning</think>"
    reasoning, caption = _extract_think_block(text)
    assert reasoning == "reasoning"
    assert caption == ""


# ---------------------------------------------------------------------------
# _extract_think_block — Case 3: only <think> present (truncated)
# ---------------------------------------------------------------------------

def test_case3_only_open_tag(caplog):
    import logging
    text = "preamble<think>incomplete reasoning cut off"
    with caplog.at_level(logging.WARNING, logger="worker.video.vlm_backend"):
        reasoning, caption = _extract_think_block(text)
    assert reasoning == "incomplete reasoning cut off"
    assert caption == "preamble"
    assert "truncated" in caplog.text.lower() or "max_tokens" in caplog.text.lower()


def test_case3_no_preamble(caplog):
    import logging
    text = "<think>only reasoning, no close tag"
    with caplog.at_level(logging.WARNING):
        reasoning, caption = _extract_think_block(text)
    assert reasoning == "only reasoning, no close tag"
    assert caption == ""


# ---------------------------------------------------------------------------
# _extract_think_block — no tags at all
# ---------------------------------------------------------------------------

def test_no_tags_returns_none_sentinel():
    """No think tags → caption is None (sentinel meaning 'use original text')."""
    text = "plain response with no think tags"
    reasoning, caption = _extract_think_block(text)
    assert reasoning == ""
    assert caption is None  # sentinel: caller should use original text, not empty string


# ---------------------------------------------------------------------------
# _extract_think_block — malformed ordering: </think> before <think>
# ---------------------------------------------------------------------------

def test_malformed_close_before_open_treated_as_truncated(caplog):
    """When </think> appears only before <think>, treat as truncated (Case 3 fallback)."""
    import logging
    text = "</think>some text<think>reasoning starts here"
    with caplog.at_level(logging.WARNING, logger="worker.video.vlm_backend"):
        reasoning, caption = _extract_think_block(text)
    # caption must NOT be None — tags were present, just malformed
    assert caption is not None
    assert "</think>" not in (caption or "")
    assert "malformed" in caplog.text.lower() or "truncated" in caplog.text.lower()


def test_malformed_does_not_return_none_sentinel(caplog):
    """Malformed ordering must never return None sentinel (would cause caller to use original text)."""
    import logging
    text = "</think>preamble<think>reasoning"
    with caplog.at_level(logging.WARNING):
        _, caption = _extract_think_block(text)
    assert caption is not None, "None sentinel is only for 'no think tags present'"


# ---------------------------------------------------------------------------
# _parse_vlm_response — integration
# ---------------------------------------------------------------------------

def test_parse_valid_json_after_think():
    text = '<think>reasoning</think>{"caption": "scene", "events": [], "entities": [], "notes": []}'
    result = _parse_vlm_response(text)
    assert result["caption"] == "scene"
    assert result["reasoning"] == "reasoning"


def test_parse_case2_think():
    text = 'some reasoning</think>{"caption": "hello", "events": [], "entities": [], "notes": []}'
    result = _parse_vlm_response(text)
    assert result["caption"] == "hello"
    assert result["reasoning"] == "some reasoning"


def test_parse_strips_answer_wrapper():
    text = "<think>r</think><answer>{\"caption\": \"c\", \"events\": [], \"entities\": [], \"notes\": []}</answer>"
    result = _parse_vlm_response(text)
    assert result["caption"] == "c"


def test_parse_empty_string():
    result = _parse_vlm_response("")
    assert result["caption"] == ""
    assert result["events"] == []


# ---------------------------------------------------------------------------
# Regression: empty caption_part must NOT fall back to original text
# (would reintroduce think block / reasoning into the caption)
# ---------------------------------------------------------------------------

def test_parse_case2_empty_caption_no_fallback():
    """Case 2 with nothing after </think> must yield empty caption, not the full text."""
    text = "some reasoning</think>"
    result = _parse_vlm_response(text)
    assert "</think>" not in result["caption"]
    assert "some reasoning" not in result["caption"]
    assert result["reasoning"] == "some reasoning"


def test_parse_case3_empty_preamble_no_fallback(caplog):
    """Case 3 with nothing before <think> must yield empty caption, not the full text."""
    import logging
    text = "<think>incomplete reasoning"
    with caplog.at_level(logging.WARNING):
        result = _parse_vlm_response(text)
    assert "<think>" not in result["caption"]
    assert "incomplete reasoning" not in result["caption"]
    assert result["reasoning"] == "incomplete reasoning"


def test_parse_no_think_tags_uses_full_text():
    """When no think tags are present, caption_part is None and full text is parsed."""
    text = '{"caption": "plain", "events": [], "entities": [], "notes": []}'
    result = _parse_vlm_response(text)
    assert result["caption"] == "plain"
    assert result.get("reasoning", "") == ""

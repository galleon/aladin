"""translate_text tool – wraps the translation service for agent use."""

from __future__ import annotations

import asyncio
import structlog
from langchain_core.tools import tool

logger = structlog.get_logger()


@tool
def translate_text(
    text: str,
    target_language: str,
    llm_model: str = "",
    simplified: bool = False,
    source_language: str = "auto",
) -> str:
    """Translate text into the specified target language.

    Args:
        text: The text to translate.
        target_language: Target language code (e.g. "de", "fr", "it").
        llm_model: LLM model to use for translation.
        simplified: Whether to use simplified language mode.
        source_language: Source language code or "auto" for auto-detect.

    Returns:
        The translated text.
    """
    from ..services.translation_service import translation_service, SUPPORTED_LANGUAGES

    if not target_language:
        return "Error: target_language is required."

    if target_language not in SUPPORTED_LANGUAGES:
        return f"Error: unsupported target language '{target_language}'."

    try:
        # Build a minimal Agent-like object for translation_service
        class _MinimalAgent:
            def __init__(self, model: str):
                self.id = 0
                self.llm_model = model or "gpt-4"
                self.temperature = 0.3
                self.max_tokens = 4096
                self.system_prompt = None

        agent = _MinimalAgent(llm_model)

        # translation_service.translate_text is async, run in event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    translation_service.translate_text(
                        text=text,
                        target_language=target_language,
                        agent=agent,
                        simplified=simplified,
                        source_language=source_language,
                    ),
                ).result()
        else:
            result = asyncio.run(
                translation_service.translate_text(
                    text=text,
                    target_language=target_language,
                    agent=agent,
                    simplified=simplified,
                    source_language=source_language,
                )
            )

        translated = result.get("translated_text", "")
        logger.info(
            "translate_text tool completed",
            target_language=target_language,
            output_length=len(translated),
        )
        return translated

    except Exception as e:
        logger.error("translate_text tool failed", error=str(e))
        return f"Translation failed: {e}"

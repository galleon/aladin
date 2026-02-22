"""Translation service for text and document translation."""

from __future__ import annotations

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Any
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from ..models import Agent

logger = structlog.get_logger()

# Language mappings
# Core languages for the translation app: German, Italian, French, English, Portuguese,
# Albanian, Spanish, Ukrainian, Russian, Serbo-Croatian, and Turkish
# Plus additional commonly supported languages
SUPPORTED_LANGUAGES = {
    # Core languages (as specified)
    "en": "English",
    "de": "German",
    "it": "Italian",
    "fr": "French",
    "pt": "Portuguese",
    "sq": "Albanian",
    "es": "Spanish",
    "uk": "Ukrainian",
    "ru": "Russian",
    "sh": "Serbo-Croatian",  # Also supports hr (Croatian), sr (Serbian), bs (Bosnian)
    "tr": "Turkish",
    # Additional commonly supported languages
    "nl": "Dutch",
    "pl": "Polish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "th": "Thai",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "ro": "Romanian",
    "hu": "Hungarian",
    # Additional European languages
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "bs": "Bosnian",
    "mk": "Macedonian",
    "sl": "Slovenian",
    "sk": "Slovak",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ga": "Irish",
    "mt": "Maltese",
    "is": "Icelandic",
    # Additional Asian languages
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "th": "Thai",
    "vi": "Vietnamese",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
    "fa": "Persian",
    # Additional languages
    "af": "Afrikaans",
    "sw": "Swahili",
    "zu": "Zulu",
    "am": "Amharic",
    "eu": "Basque",
    "ca": "Catalan",
    "gl": "Galician",
    "cy": "Welsh",
    "br": "Breton",
}


class TranslationService:
    """Service for translating text and documents."""

    def __init__(self):
        self._llm_cache: dict[str, Any] = {}

    def get_llm(self, model: str, temperature: float = 0.3, max_tokens: int = 4096):
        """Get or create an LLM instance for translation."""
        cache_key = f"trans_{model}_{temperature}_{max_tokens}"

        if cache_key not in self._llm_cache:
            # Use longer timeout for models that might take time (e.g., "thinking" models)
            # Default to 300 seconds (5 minutes) for subtitle translation
            timeout_value = 300.0 if "thinking" in model.lower() else 120.0
            self._llm_cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_API_BASE,
                timeout=timeout_value,
                max_retries=2,  # Retry up to 2 times on failure
            )
            logger.info(
                "Created translation LLM instance",
                model=model,
                temperature=temperature,
                base_url=settings.LLM_API_BASE,
            )

        return self._llm_cache[cache_key]

    def clear_llm_cache(self, model: str | None = None):
        """Clear LLM cache, optionally for a specific model."""
        if model:
            # Clear cache for specific model
            keys_to_remove = [
                k for k in self._llm_cache.keys() if k.startswith(f"trans_{model}_")
            ]
            for key in keys_to_remove:
                del self._llm_cache[key]
            logger.info("Cleared LLM cache for model", model=model)
        else:
            # Clear all cache
            self._llm_cache.clear()
            logger.info("Cleared all LLM cache")

    def _clean_translation_output(self, text: str, srt_mode: bool = False) -> str:
        """Clean translation output to remove reasoning and meta-commentary.
        When srt_mode=True, only strip leading preamble; do not split/rejoin by '. '
        (that would corrupt SRT structure and drop most content).
        """
        if not text:
            return text

        # Remove common reasoning markers and extract text after them
        markers_to_remove = [
            "assistantfinal",
            "translation:",
            "Translation:",
            "The translation is:",
            "Here is the translation:",
            "Final translation:",
            "final translation:",
        ]

        # First, try to extract text after markers
        text_lower = text.lower()
        for marker in markers_to_remove:
            if marker.lower() in text_lower:
                idx = text_lower.find(marker.lower())
                # Extract text after the marker
                text = text[idx + len(marker) :].strip()
                text_lower = text.lower()
                break

        if srt_mode:
            return text.strip()

        # Split by lines and filter out reasoning
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            line_lower = line_stripped.lower()

            # Skip lines that are clearly reasoning or meta-commentary
            reasoning_phrases = [
                "we need to translate",
                "the user wrote",
                "the user says",
                "so translation",
                "provide only",
                "that is",
                "but we",
                "so we",
                "we can",
                "we must",
                "the instruction says",
                "but the",
                "however,",
                "note:",
                "important:",
            ]

            if any(phrase in line_lower for phrase in reasoning_phrases):
                continue

            # Remove any remaining markers from the line
            for marker in markers_to_remove:
                if marker.lower() in line_lower:
                    idx = line_lower.find(marker.lower())
                    line_stripped = line_stripped[idx + len(marker) :].strip()
                    break

            if line_stripped:
                cleaned_lines.append(line_stripped)

        result = "\n".join(cleaned_lines).strip()

        # If result is still very long or contains multiple paragraphs,
        # try to extract just the translation part
        if len(result) > 300 or "\n\n" in result:
            # Look for the last substantial sentence/paragraph that doesn't contain reasoning
            sentences = result.split(". ")
            translation_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Skip sentences with reasoning patterns
                if not any(
                    phrase in sentence_lower
                    for phrase in [
                        "we need",
                        "the user",
                        "so translation",
                        "but we",
                        "provide only",
                        "the instruction",
                    ]
                ):
                    translation_sentences.append(sentence)

            if translation_sentences:
                result = ". ".join(translation_sentences).strip()
                if not result.endswith(".") and len(translation_sentences) > 1:
                    result += "."

        # Final cleanup: remove any leading/trailing punctuation artifacts
        result = result.strip(".,!?;: \n\t")

        return result

    def _format_custom_prompt(
        self, template: str, target_language: str, simplified: bool
    ) -> str:
        """Format custom prompt with placeholders {target_language}, {simplified}."""
        lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
        try:
            return template.format(
                target_language=lang_name,
                simplified=simplified,
            )
        except (KeyError, ValueError):
            return template

    def get_system_prompt(
        self,
        target_language: str,
        simplified: bool = False,
        agent: Agent | None = None,
    ) -> str:
        """Generate system prompt for translation."""
        lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)

        if agent and agent.system_prompt and str(agent.system_prompt).strip():
            return self._format_custom_prompt(
                agent.system_prompt, target_language, simplified
            )

        if simplified:
            return f"""You are a professional translator specializing in simplified language. Translate the given text into {lang_name} using simple, easy-to-understand language.

CRITICAL RULES:
1. Output ONLY the translated text - nothing else
2. Do NOT include any explanations, reasoning, or meta-commentary
3. Do NOT show your thought process
4. Do NOT include phrases like "assistantfinal" or "translation:"
5. Use short, simple sentences
6. Use common, everyday words
7. Avoid jargon and technical terms
8. Preserve formatting and structure

Your response must be ONLY the translation, with no additional text before or after it."""
        else:
            return f"""You are a professional translator. Translate the given text into {lang_name}.

CRITICAL RULES:
1. Output ONLY the translated text - nothing else
2. Do NOT include any explanations, reasoning, or meta-commentary
3. Do NOT show your thought process
4. Do NOT include phrases like "assistantfinal" or "translation:"
5. If the text contains multiple languages, translate all parts to {lang_name}
6. Preserve formatting, punctuation, and structure
7. Maintain the original tone and style

Your response must be ONLY the translation, with no additional text before or after it."""

    async def translate_text(
        self,
        text: str,
        target_language: str,
        agent: Agent,
        simplified: bool = False,
        source_language: str = "auto",
    ) -> dict[str, Any]:
        """Translate a piece of text."""
        llm = self.get_llm(
            agent.llm_model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )

        system_prompt = self.get_system_prompt(target_language, simplified, agent)

        # Add source language hint if specified
        user_message = text
        if source_language != "auto":
            source_name = SUPPORTED_LANGUAGES.get(source_language, source_language)
            user_message = f"[Source language: {source_name}]\n\n{text}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        try:
            response = llm.invoke(messages)

            # Extract content - handle different response types
            if hasattr(response, "content"):
                translated_text = response.content
            elif hasattr(response, "text"):
                translated_text = response.text
            elif isinstance(response, str):
                translated_text = response
            else:
                translated_text = str(response)

            # Clean up the translation - remove any reasoning or meta-commentary
            translated_text = self._clean_translation_output(translated_text)

            # Extract token usage if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                elif hasattr(usage, "input_tokens"):
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)

            logger.info(
                "Text translation completed",
                target_language=target_language,
                simplified=simplified,
                input_length=len(text),
                output_length=len(translated_text),
            )

            return {
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language,
                "simplified": simplified,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Translation error",
                agent_id=agent.id,
                model=agent.llm_model,
                error=error_msg,
                base_url=settings.LLM_API_BASE,
            )

            # Clear the cached LLM instance on connection errors to force recreation
            if "Connection" in error_msg or "timeout" in error_msg.lower():
                self.clear_llm_cache(agent.llm_model)
                raise ValueError(
                    f"Unable to connect to the model endpoint. "
                    f"Please check your network connection and endpoint configuration."
                )

            # Check for specific error types
            if "404" in error_msg or "Not Found" in error_msg:
                raise ValueError(
                    f"Model '{agent.llm_model}' is not available at the configured endpoint. "
                    f"Please check your model configuration or contact your administrator."
                )
            elif "401" in error_msg or "Unauthorized" in error_msg:
                raise ValueError(
                    "Authentication failed. Please check your API key configuration."
                )
            else:
                raise ValueError(
                    f"Translation failed: {error_msg}. "
                    f"Please try again or contact support if the issue persists."
                )

    async def translate_markdown(
        self,
        markdown: str,
        target_language: str,
        agent: Agent,
        simplified: bool = False,
    ) -> dict[str, Any]:
        """Translate markdown content, preserving formatting."""
        llm = self.get_llm(
            agent.llm_model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )

        lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)

        if agent.system_prompt and str(agent.system_prompt).strip():
            system_prompt = self._format_custom_prompt(
                agent.system_prompt, target_language, simplified
            )
        elif simplified:
            system_prompt = f"""You are an expert translator specializing in simplified language.
Translate the following markdown document into {lang_name} using simple, easy-to-understand language.

IMPORTANT: Preserve all markdown formatting exactly:
- Keep headers (#, ##, ###)
- Keep lists (-, *, 1., 2.)
- Keep bold (**text**) and italic (*text*)
- Keep code blocks (```)
- Keep links [text](url)
- Keep tables
- Keep images ![alt](src)

Only translate the text content, not the markdown syntax or URLs.
Use simple sentences and common words suitable for all readers."""
        else:
            system_prompt = f"""You are an expert translator with deep knowledge of {lang_name}.
Translate the following markdown document into {lang_name}.

IMPORTANT: Preserve all markdown formatting exactly:
- Keep headers (#, ##, ###)
- Keep lists (-, *, 1., 2.)
- Keep bold (**text**) and italic (*text*)
- Keep code blocks (```)
- Keep links [text](url)
- Keep tables
- Keep images ![alt](src)

Only translate the text content, not the markdown syntax or URLs.
Maintain the original tone, style, and technical accuracy."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=markdown),
        ]

        try:
            response = llm.invoke(messages)

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                input_tokens = response.usage_metadata.get("input_tokens", 0)
                output_tokens = response.usage_metadata.get("output_tokens", 0)

            logger.info(
                "Markdown translation completed",
                target_language=target_language,
                simplified=simplified,
            )

            return {
                "translated_markdown": response.content,
                "target_language": target_language,
                "simplified": simplified,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except Exception as e:
            logger.error("Markdown translation failed", error=str(e))
            raise


# Singleton instance
translation_service = TranslationService()

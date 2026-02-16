"""Video transcription service using external Whisper API."""

import json
import os
import re
import tempfile
import subprocess
from pathlib import Path
from typing import BinaryIO
import structlog
import httpx

from ..config import settings

# OpenAI Whisper API limit (25 MB). Use 24 MB to leave headroom.
OPENAI_WHISPER_MAX_BYTES = 24 * 1024 * 1024
# At 16 kHz mono s16le: 32_000 bytes/sec → ~600 s ≈ 19 MB per chunk
OPENAI_WHISPER_CHUNK_DURATION_SEC = 600

# Subtitle translation: max segments per LLM call to stay under context limit
SUBTITLE_TRANSLATION_BATCH_SIZE = 80
from .translation_service import TranslationService, SUPPORTED_LANGUAGES

logger = structlog.get_logger()
translation_service = TranslationService()


class VideoTranscriptionService:
    """Service for transcribing videos and adding subtitles using external Whisper API."""

    def __init__(self):
        self.whisper_api_base = settings.WHISPER_API_BASE
        self.whisper_api_key = settings.WHISPER_API_KEY
        self.upload_dir = Path("/app/uploads/videos")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Detect if using OpenAI's API
        self.is_openai_api = self.whisper_api_base and (
            "api.openai.com" in self.whisper_api_base
            or self.whisper_api_base.rstrip("/").endswith("/v1")
        )

        if not self.whisper_api_base:
            logger.warning(
                "WHISPER_API_BASE not configured - video transcription will be disabled"
            )
        elif self.is_openai_api:
            logger.info("Using OpenAI Whisper API")

    def transcribe_video(
        self,
        video_file: BinaryIO,
        language: str | None = None,
        model_size: str | None = None,
    ) -> dict:
        """
        Transcribe a video file using external Whisper API.

        Args:
            video_file: Video file to transcribe
            language: Optional language code (e.g., 'en', 'fr'). If None, auto-detect.
            model_size: Whisper model size (tiny, base, small, medium, large)

        Returns:
            dict with 'transcript' (text) and 'segments' (list of dicts with start, end, text)
        """
        # Check if Whisper API is configured
        if not self.whisper_api_base:
            raise ValueError(
                "WHISPER_API_BASE is not configured. Video transcription requires an external Whisper API."
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

        try:
            # Use provided model_size or fall back to default
            whisper_model = model_size or "base"

            # Always use external Whisper API
            return self._transcribe_with_remote_api(
                temp_video_path, whisper_model, language
            )
        except Exception as e:
            logger.error("Transcription failed", error=str(e), exc_info=True)
            raise
        finally:
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)

    def _transcribe_with_remote_api(
        self, video_path: str, model_size: str, language: str | None
    ) -> dict:
        """Transcribe using external Whisper API (supports OpenAI API or custom service)."""
        if self.is_openai_api:
            return self._transcribe_with_openai_api(video_path, language)
        else:
            return self._transcribe_with_custom_api(video_path, model_size, language)

    def _extract_audio_to_wav(self, video_path: str, sample_rate: int = 16000) -> str:
        """Extract audio from video to WAV format for better transcription accuracy."""
        wav_path = video_path.replace(".mp4", ".wav").replace(".mov", ".wav")
        if not wav_path.endswith(".wav"):
            wav_path = f"{video_path}.wav"

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-i",
            video_path,
            "-vn",  # No video
            "-ac",
            "1",  # Mono
            "-ar",
            str(sample_rate),  # Sample rate
            "-c:a",
            "pcm_s16le",  # PCM 16-bit little-endian
            wav_path,
        ]

        logger.info("Extracting audio to WAV", video_path=video_path, wav_path=wav_path)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        return wav_path

    def _get_audio_duration_sec(self, wav_path: str) -> float:
        """Get duration of an audio file in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _extract_wav_segment(
        self, wav_path: str, start_sec: float, end_sec: float, out_path: str
    ) -> None:
        """Extract a time range from WAV to a new file using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-ss",
            str(start_sec),
            "-to",
            str(end_sec),
            "-c:a",
            "pcm_s16le",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _post_openai_whisper_chunk(
        self, audio_path: str, language: str | None
    ) -> tuple[str, list[dict], str]:
        """POST one audio file to OpenAI Whisper API; return (transcript, segments, language)."""
        base_url = self.whisper_api_base.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        url = f"{base_url}/audio/transcriptions"
        headers = {}
        if self.whisper_api_key:
            headers["Authorization"] = f"Bearer {self.whisper_api_key}"

        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0",
            }
            if language:
                data["language"] = language
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                result = response.json()

        transcript_text = result.get("text", "")
        detected_language = result.get("language", language or "unknown")
        segments = []
        if "segments" in result and result["segments"]:
            for seg in result["segments"]:
                segments.append(
                    {
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", seg.get("start", 0.0))),
                        "text": (seg.get("text") or "").strip(),
                    }
                )
        elif "words" in result and result["words"]:
            words = result["words"]
            current_segment = {"start": None, "end": None, "text": []}
            segment_max_duration = 5.0
            segment_start_time = None
            for i, word_info in enumerate(words):
                word_start = float(word_info.get("start", 0))
                word_end = float(word_info.get("end", 0))
                word_text = word_info.get("word", "")
                if current_segment["start"] is None:
                    current_segment["start"] = word_start
                    segment_start_time = word_start
                current_segment["end"] = word_end
                current_segment["text"].append(word_text)
                segment_duration = word_end - segment_start_time
                is_sentence_end = word_text.rstrip(".,!?;:") != word_text
                is_last_word = i == len(words) - 1
                if (
                    segment_duration >= segment_max_duration
                    or is_sentence_end
                    or is_last_word
                ):
                    if current_segment["start"] is not None and current_segment["text"]:
                        segments.append(
                            {
                                "start": current_segment["start"],
                                "end": current_segment["end"],
                                "text": " ".join(current_segment["text"]).strip(),
                            }
                        )
                    if not is_last_word:
                        current_segment = {"start": None, "end": None, "text": []}
                        segment_start_time = None
        else:
            duration = result.get("duration", 0)
            segments.append(
                {"start": 0.0, "end": float(duration), "text": transcript_text}
            )
        return transcript_text, segments, detected_language

    def _transcribe_with_openai_api(
        self, video_path: str, language: str | None
    ) -> dict:
        """Transcribe using OpenAI's Whisper API with audio extraction.
        Files larger than 25 MB are split into chunks and merged.
        """
        logger.info(
            "Calling OpenAI Whisper API",
            api_base=self.whisper_api_base,
            language=language,
        )

        wav_path = None
        try:
            wav_path = self._extract_audio_to_wav(video_path)
            wav_size = os.path.getsize(wav_path)

            if wav_size <= OPENAI_WHISPER_MAX_BYTES:
                # Single request
                transcript_text, segments, detected_language = (
                    self._post_openai_whisper_chunk(wav_path, language)
                )
                logger.info(
                    "OpenAI transcription completed",
                    language=detected_language,
                    segments_count=len(segments),
                )
                return {
                    "transcript": transcript_text,
                    "segments": segments,
                    "language": detected_language,
                    "language_probability": 1.0,
                }

            # Chunked: file exceeds 25 MB limit
            duration_sec = self._get_audio_duration_sec(wav_path)
            all_segments = []
            all_texts = []
            detected_language = language or "unknown"
            chunk_num = 0
            start = 0.0
            while start < duration_sec:
                end = min(start + OPENAI_WHISPER_CHUNK_DURATION_SEC, duration_sec)
                chunk_num += 1
                fd, chunk_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    self._extract_wav_segment(wav_path, start, end, chunk_path)
                    logger.info(
                        "Transcribing chunk",
                        chunk=chunk_num,
                        start_sec=start,
                        end_sec=end,
                    )
                    text, segments, detected_language = self._post_openai_whisper_chunk(
                        chunk_path, language
                    )
                    for seg in segments:
                        all_segments.append(
                            {
                                "start": seg["start"] + start,
                                "end": seg["end"] + start,
                                "text": seg["text"],
                            }
                        )
                    if text.strip():
                        all_texts.append(text)
                finally:
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
                start = end

            full_transcript = " ".join(all_texts).strip()
            logger.info(
                "OpenAI chunked transcription completed",
                language=detected_language,
                chunks=chunk_num,
                segments_count=len(all_segments),
            )
            return {
                "transcript": full_transcript,
                "segments": all_segments,
                "language": detected_language,
                "language_probability": 1.0,
            }
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    def _transcribe_with_custom_api(
        self, video_path: str, model_size: str, language: str | None
    ) -> dict:
        """Transcribe using custom Whisper API service."""
        logger.info(
            "Calling custom Whisper API",
            api_base=self.whisper_api_base,
            model_size=model_size,
            language=language,
        )

        url = f"{self.whisper_api_base.rstrip('/')}/transcribe"
        headers = {}
        if self.whisper_api_key:
            headers["Authorization"] = f"Bearer {self.whisper_api_key}"

        with open(video_path, "rb") as video_file:
            files = {"file": ("video.mp4", video_file, "video/mp4")}
            data = {"model_size": model_size}
            if language:
                data["language"] = language

            with httpx.Client(
                timeout=300.0
            ) as client:  # 5 minute timeout for large videos
                response = client.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                result = response.json()

        logger.info(
            "Custom API transcription completed",
            language=result.get("language"),
            segments_count=len(result.get("segments", [])),
        )

        return {
            "transcript": result.get("transcript", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", language or "unknown"),
            "language_probability": result.get("language_probability", 1.0),
        }

    def add_subtitles_to_video(
        self, video_path: str, subtitles: list[dict], output_path: str
    ) -> str:
        """
        Add subtitles to a video file using FFmpeg.

        Args:
            video_path: Path to input video
            subtitles: List of subtitle dicts with 'start', 'end', 'text'
            output_path: Path to save output video

        Returns:
            Path to output video with subtitles
        """
        # Create SRT file
        srt_path = output_path.replace(".mp4", ".srt")
        self._create_srt_file(srt_path, subtitles)

        try:
            # Use FFmpeg to embed subtitles as burned-in text
            # Use absolute path for SRT file to avoid path issues
            srt_path_abs = os.path.abspath(srt_path)

            # Use subtitles filter with styling
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-i",
                video_path,
                "-vf",
                f"subtitles={srt_path_abs}:force_style='FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Alignment=2'",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-c:a",
                "copy",
                output_path,
            ]

            logger.info(
                "Adding subtitles to video",
                output_path=output_path,
                segments_count=len(subtitles),
            )
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger.info("Subtitles added successfully", output_path=output_path)
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(
                "Failed to add subtitles",
                error=e.stderr,
                returncode=e.returncode,
                stdout=e.stdout,
            )
            raise
        finally:
            # Clean up SRT file
            if os.path.exists(srt_path):
                os.unlink(srt_path)

    def save_srt_file(self, segments: list[dict], srt_path: str) -> None:
        """Save segments to an SRT file (public method for saving intermediate files)."""
        self._create_srt_file(srt_path, segments)

    def _create_srt_file(self, srt_path: str, subtitles: list[dict]) -> None:
        """Create an SRT subtitle file from segments."""
        import re

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(subtitles, 1):
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", start))
                text = (segment.get("text") or "").strip()
                # Normalize whitespace
                text = re.sub(r"\s+", " ", text)

                start_time = self._format_srt_time(start)
                end_time = self._format_srt_time(end)

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)."""
        if seconds < 0:
            seconds = 0.0
        ms_total = int(round(seconds * 1000.0))
        hh = ms_total // 3_600_000
        ms_total -= hh * 3_600_000
        mm = ms_total // 60_000
        ms_total -= mm * 60_000
        ss = ms_total // 1_000
        ms = ms_total - ss * 1_000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    def _segments_to_srt_text(self, segments: list[dict]) -> str:
        """Build SRT-formatted string from segment dicts (start, end, text)."""
        lines = []
        for i, segment in enumerate(segments, start=1):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            text = (segment.get("text") or "").strip()
            text = re.sub(r"\s+", " ", text)
            start_time = self._format_srt_time(start)
            end_time = self._format_srt_time(end)
            lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
        return "\n".join(lines)

    def _group_segments_into_sentences(self, segments: list[dict]) -> list[list[dict]]:
        """
        Group SRT segments into sentence units. Buffer segments until a full
        sentence is formed (text ends with . ! or ?).
        """
        if not segments:
            return []
        groups: list[list[dict]] = []
        buffer: list[dict] = []
        for seg in segments:
            buffer.append(seg)
            text = (seg.get("text") or "").strip()
            text = re.sub(r"\s+", " ", text)
            if text and text[-1] in ".!?":
                groups.append(buffer[:])
                buffer = []
        if buffer:
            groups.append(buffer)
        return groups

    def _redistribute_times(
        self, segment_group: list[dict], translated_strings: list[str]
    ) -> list[dict]:
        """
        Redistribute the original total duration of the segment group across
        the new translated segments proportionally to character length.
        Returns segment dicts with start, end, text (translated).
        """
        if not segment_group or not translated_strings:
            return []
        total_duration = float(segment_group[-1].get("end", 0)) - float(
            segment_group[0].get("start", 0)
        )
        start_base = float(segment_group[0].get("start", 0))
        total_chars = sum(len(s) for s in translated_strings)
        if total_chars <= 0:
            return [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": translated_strings[i]
                    if i < len(translated_strings)
                    else seg.get("text", ""),
                }
                for i, seg in enumerate(segment_group)
            ]
        result = []
        t = start_base
        for i, text in enumerate(translated_strings):
            ratio = len(text) / total_chars
            duration = total_duration * ratio
            result.append(
                {
                    "start": round(t, 3),
                    "end": round(t + duration, 3),
                    "text": text.strip(),
                }
            )
            t += duration
        return result

    def translate_subtitles(
        self,
        segments: list[dict],
        source_language: str,
        target_language: str,
        translation_agent=None,
    ) -> list[dict]:
        """
        Translate subtitle segments from source language to target language.
        Uses SRT-based translation approach (similar to standalone script) for better reliability.

        Args:
            segments: List of subtitle dicts with 'start', 'end', 'text'
            source_language: Source language code (e.g., 'en', 'fr')
            target_language: Target language code (e.g., 'es', 'de')
            translation_agent: Optional Agent object (translation agent) to use for translation.
                              If not provided, translation will be skipped.

        Returns:
            List of translated subtitle dicts with same structure
        """
        if not segments:
            return segments

        # If no translation agent provided, skip translation
        if not translation_agent:
            logger.warning("No translation agent provided, skipping translation")
            return segments

        # Check if agent has required LLM configuration
        if not translation_agent.llm_model:
            logger.warning(
                "Translation agent has no LLM model configured, skipping translation",
                agent_id=translation_agent.id,
            )
            return segments
        if not segments:
            return segments

        logger.info(
            "Translating subtitles",
            source_language=source_language,
            target_language=target_language,
            segments_count=len(segments),
            translation_agent_id=translation_agent.id,
            translation_agent_name=translation_agent.name,
        )

        # Get source and target language names
        source_lang_name = SUPPORTED_LANGUAGES.get(
            source_language.lower(), source_language
        )
        target_lang_name = SUPPORTED_LANGUAGES.get(
            target_language.lower(), target_language
        )

        import re
        from langchain_core.messages import HumanMessage, SystemMessage

        # Use translation agent's configuration
        model = translation_agent.llm_model
        temperature = (
            translation_agent.temperature
            if translation_agent.temperature is not None
            else 0.1
        )
        max_tokens = (
            translation_agent.max_tokens
            if translation_agent.max_tokens is not None
            else 4096
        )
        # Ensure max_tokens is at least 1 (avoids invalid API requests)
        max_tokens = max(1, max_tokens)

        logger.info(
            "Using translation agent configuration",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            llm = translation_service.get_llm(
                model=model, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(
                "Failed to create LLM instance for translation",
                model=model,
                error=str(e),
                translation_agent_id=translation_agent.id,
            )
            raise RuntimeError(
                f"Translation agent '{translation_agent.name}' has invalid LLM model '{model}'. "
                f"Please update the agent's model configuration. Error: {str(e)}"
            )

        # Sentence-based sliding window: lean system prompt for JSON-only output
        src_hint = (
            f"Source language: {source_lang_name}."
            if source_language
            else "Detect the source language."
        )
        system_prompt = (
            "You are a subtitle translator. Output ONLY a valid JSON array of strings.\n"
            f"{src_hint} Target language: {target_lang_name}.\n"
            "Rules: Return a JSON array with exactly N strings (N = number of segments in the task). "
            "Each string is the translation of the corresponding segment. No commentary, no markdown, no explanation."
        )

        # Group segments into sentences (buffer until . ! ?)
        sentence_groups = self._group_segments_into_sentences(segments)
        logger.info(
            "Sentence-based translation",
            total_segments=len(segments),
            sentence_groups=len(sentence_groups),
        )

        try:
            all_translated = []
            previous_context = ""
            for group_idx, group in enumerate(sentence_groups):
                current_sentence = " ".join(
                    (seg.get("text") or "").strip() for seg in group
                ).strip()
                current_sentence = re.sub(r"\s+", " ", current_sentence)
                if not current_sentence:
                    all_translated.extend(group)
                    continue

                # Lean prompt: Context | Task
                if previous_context:
                    user_message = (
                        f"Context: {previous_context} | Task: Translate to {target_lang_name} "
                        f"(return JSON array of {len(group)} strings): {current_sentence}"
                    )
                else:
                    user_message = (
                        f"Task: Translate to {target_lang_name} "
                        f"(return JSON array of {len(group)} strings): {current_sentence}"
                    )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                logger.info(
                    "Translating sentence group",
                    group_index=group_idx + 1,
                    total_groups=len(sentence_groups),
                    segment_count=len(group),
                )
                response = llm.invoke(messages)
                raw = (
                    response.content
                    if hasattr(response, "content")
                    else (response.text if hasattr(response, "text") else str(response))
                )
                raw = raw.strip()
                # Strip markdown code block if present
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```\s*$", "", raw)

                try:
                    translated_strings = json.loads(raw)
                    if not isinstance(translated_strings, list):
                        translated_strings = [str(translated_strings)]
                    translated_strings = [str(s).strip() for s in translated_strings]
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to parse JSON from LLM, using original for group",
                        group_index=group_idx,
                        raw_preview=raw[:200],
                    )
                    all_translated.extend(group)
                    previous_context = ""  # Don't pass untranslated as context
                    continue

                # Ensure we have exactly N strings; pad or trim to match group size
                n = len(group)
                if len(translated_strings) < n:
                    translated_strings.extend(
                        (
                            group[i].get("text", "")
                            for i in range(len(translated_strings), n)
                        )
                    )
                elif len(translated_strings) > n:
                    translated_strings = translated_strings[:n]

                # Time redistribution: scale display time by character length
                group_translated = self._redistribute_times(group, translated_strings)
                all_translated.extend(group_translated)
                previous_context = " ".join(translated_strings)

            if len(all_translated) != len(segments):
                logger.warning(
                    "Translated segment count mismatch, using original for missing",
                    original_count=len(segments),
                    translated_count=len(all_translated),
                )
                return segments

            logger.info(
                "Subtitles translated successfully (sentence-based)",
                source_language=source_language,
                target_language=target_language,
                translated_count=len(all_translated),
                sentence_groups=len(sentence_groups),
            )
            return all_translated
        except Exception as e:
            logger.error(
                "Failed to translate subtitles",
                error=str(e),
                source_language=source_language,
                target_language=target_language,
                translation_agent_id=translation_agent.id,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to translate subtitles from {source_language} to {target_language}: {str(e)}"
            ) from e

    def _parse_srt_to_segments(
        self, srt_text: str, original_segments: list[dict]
    ) -> list[dict]:
        """Parse SRT text back to segment dicts, preserving timestamps from original."""
        import re

        blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.MULTILINE)
        translated_segments = []

        for i, block in enumerate(blocks):
            if i >= len(original_segments):
                break

            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2:
                # Use original if we can't parse
                translated_segments.append(original_segments[i])
                continue

            # Find timestamp line
            time_line = None
            for line in lines:
                if "-->" in line:
                    time_line = line
                    break

            if not time_line:
                # Use original if no timestamp found
                translated_segments.append(original_segments[i])
                continue

            # Extract text (everything after timestamp line)
            time_idx = lines.index(time_line) if time_line in lines else 0
            text_lines = lines[time_idx + 1 :]
            translated_text = " ".join(text_lines).strip()

            # If no translated text found, use original
            if not translated_text:
                logger.warning(
                    "No translated text found for segment, using original",
                    segment_index=i,
                    block_preview=block[:100],
                )
                translated_segments.append(original_segments[i])
                continue

            # Use original timestamps but translated text
            translated_segments.append(
                {
                    "start": original_segments[i].get("start", 0.0),
                    "end": original_segments[i].get("end", 0.0),
                    "text": translated_text
                    if translated_text
                    else original_segments[i].get("text", ""),
                }
            )

        # Fill in any missing segments with originals
        while len(translated_segments) < len(original_segments):
            translated_segments.append(original_segments[len(translated_segments)])

        return translated_segments


# Singleton instance
video_transcription_service = VideoTranscriptionService()

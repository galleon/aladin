"""
Avatar Worker — LiveKit-based live avatar agent for the Aladin platform.

This worker:
  1. Connects to a LiveKit room as a server-side participant.
  2. Wraps SoulX-LiveTalk (inference.py) to drive lip-sync video frames from text.
  3. Listens to the Aladin LLM/RAG stream (via rag_service query) and pipes
     each token chunk into the avatar synthesis pipeline.

Performance note: on DGX Spark (ARM64 / Blackwell GPU) torch.compile() is
applied to the avatar model for maximum throughput.

Environment variables expected at runtime:
  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET  (from Arq job payload)
  LLM_API_BASE, LLM_API_KEY                         (from shared config)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SoulX-LiveTalk shim
# ---------------------------------------------------------------------------

class _AvatarSynthesizer:
    """
    Thin wrapper around SoulX-LiveTalk's inference pipeline.

    SoulX-LiveTalk is expected to be installed as a Python package or placed
    on PYTHONPATH.  If it is not available the worker degrades gracefully to
    an audio-only mode (text-to-speech frames are logged instead of rendered).
    """

    def __init__(self, avatar_config: dict[str, Any]):
        self.avatar_config = avatar_config
        self._model: Any = None
        self._available = False
        self._init()

    def _init(self) -> None:
        try:
            # SoulX-LiveTalk exposes a top-level `inference` module with a
            # `LiveTalkPipeline` class.
            from inference import LiveTalkPipeline  # type: ignore[import]

            import torch

            pipeline = LiveTalkPipeline(
                video_source=self.avatar_config.get("video_source_url", ""),
                image_source=self.avatar_config.get("image_url", ""),
            )

            # Apply torch.compile when a CUDA-capable GPU is available.
            # On DGX Spark (ARM64 / Blackwell), max-autotune selects the
            # best kernel for the GPU architecture automatically.
            if torch.cuda.is_available():
                try:
                    pipeline.model = torch.compile(
                        pipeline.model,
                        backend="inductor",
                        mode="max-autotune",
                    )
                    logger.info("torch.compile applied to avatar model (Blackwell/inductor)")
                except Exception as compile_err:
                    logger.warning(
                        "torch.compile unavailable — running in eager mode: %s",
                        compile_err,
                    )

            self._model = pipeline
            self._available = True
            logger.info("SoulX-LiveTalk pipeline initialised")
        except ImportError:
            logger.warning(
                "SoulX-LiveTalk (inference) not found — avatar video synthesis disabled. "
                "Install the package and ensure `inference.py` is on PYTHONPATH."
            )

    async def synthesize_frames(self, text_chunk: str):
        """
        Yield raw video frames (bytes) for the given text chunk.

        Falls back to a no-op generator when SoulX-LiveTalk is not available.
        """
        if not self._available or self._model is None:
            logger.debug("Avatar synthesis skipped (SoulX-LiveTalk unavailable): %r", text_chunk)
            return

        loop = asyncio.get_event_loop()
        # Run the CPU/GPU-bound synthesis in a thread-pool executor so the
        # event loop is not blocked.
        frames = await loop.run_in_executor(
            None, self._model.synthesize, text_chunk
        )
        for frame in frames:
            yield frame


# ---------------------------------------------------------------------------
# ARQ task entry-point
# ---------------------------------------------------------------------------

async def run_avatar_worker(
    ctx: dict,
    agent_id: int,
    room_name: str,
    livekit_url: str,
    livekit_api_key: str,
    livekit_api_secret: str,
    avatar_config: dict | None = None,
    llm_model: str = "gpt-4",
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """
    ARQ task: spin up a LiveKit avatar agent in *room_name*.

    Steps
    -----
    1. Connect to the LiveKit room via ``livekit.rtc``.
    2. Initialise the SoulX-LiveTalk synthesizer.
    3. Subscribe to data messages from other participants (user text).
    4. For each user message: stream an LLM response, push each token to the
       avatar synthesizer, and publish the resulting video frames to the room.
    """
    avatar_config = avatar_config or {}
    logger.info(
        "Avatar worker starting",
        agent_id=agent_id,
        room_name=room_name,
        livekit_url=livekit_url,
    )

    synthesizer = _AvatarSynthesizer(avatar_config)

    try:
        import livekit.rtc as rtc  # type: ignore[import]
    except ImportError:
        logger.error(
            "livekit-rtc not installed — cannot connect to LiveKit room. "
            "Install livekit-agents[rtc] or livekit-rtc."
        )
        return

    # Build a server-side participant token for the worker itself
    try:
        from livekit.api import AccessToken, VideoGrants  # type: ignore[import]

        worker_token = (
            AccessToken(livekit_api_key, livekit_api_secret)
            .with_identity(f"avatar-worker-{agent_id}")
            .with_name("Avatar Worker")
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )
    except Exception as exc:
        logger.error("Failed to create worker LiveKit token: %s", exc)
        return

    # Connect to the room
    room = rtc.Room()
    try:
        await room.connect(livekit_url, worker_token)
        logger.info("Avatar worker connected to LiveKit room", room=room_name)
    except Exception as exc:
        logger.error("LiveKit connect failed: %s", exc)
        return

    # LLM helper — streams tokens from the Aladin LLM endpoint
    async def _stream_llm_response(user_message: str):
        """Yield text chunks from the configured LLM."""
        from openai import AsyncOpenAI  # type: ignore[import]

        client = AsyncOpenAI(
            api_key=os.environ.get("LLM_API_KEY", "sk-dummy"),
            base_url=os.environ.get("LLM_API_BASE", "http://localhost:8000/v1"),
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        async with client.chat.completions.stream(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for event in stream:
                delta = (
                    event.choices[0].delta.content
                    if event.choices and event.choices[0].delta.content
                    else ""
                )
                if delta:
                    yield delta

    # Data message handler
    async def _handle_data_message(data_packet: rtc.DataPacket):
        user_text = data_packet.data.decode("utf-8", errors="replace")
        logger.info("Avatar worker received user message", text=user_text[:80])

        full_response: list[str] = []
        async for token_chunk in _stream_llm_response(user_text):
            full_response.append(token_chunk)
            # Send text subtitle back to the room via data channel
            await room.local_participant.publish_data(
                token_chunk.encode("utf-8"),
                reliable=True,
            )
            # Drive the avatar with this chunk
            async for _frame in synthesizer.synthesize_frames(token_chunk):
                # TODO: publish video frames to the LiveKit room via
                # rtc.VideoSource + rtc.VideoTrack once SoulX-LiveTalk
                # is wired in. Frame bytes come from the synthesizer.
                pass

        logger.info(
            "Avatar response complete",
            response_length=sum(len(c) for c in full_response),
        )

    # Register the data handler and wait until the room is disconnected
    @room.on("data_received")
    def on_data(data_packet: rtc.DataPacket):
        asyncio.ensure_future(_handle_data_message(data_packet))

    # Keep the worker alive until the room closes or it is cancelled
    try:
        disconnect_event = asyncio.Event()

        @room.on("disconnected")
        def on_disconnect(_reason: str = ""):
            disconnect_event.set()

        await disconnect_event.wait()
    finally:
        await room.disconnect()
        logger.info("Avatar worker disconnected", room=room_name)

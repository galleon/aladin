"""Voice API endpoints for Speech-to-Text and Text-to-Speech."""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas import (
    VoiceTranscribeRequest,
    VoiceTranscribeResponse,
    VoiceTextToSpeechRequest,
)
from ..services.auth import get_current_active_user
from ..config import settings
import structlog
import httpx
import io

logger = structlog.get_logger()

router = APIRouter(prefix="/voice", tags=["Voice"])


@router.post("/transcribe", response_model=VoiceTranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Transcribe audio to text using Speech-to-Text API.
    
    Supports various audio formats (mp3, mp4, mpeg, mpga, m4a, wav, webm).
    Uses OpenAI-compatible API endpoint configured in STT_API_BASE.
    """
    logger.info(
        "Transcribing audio",
        user_id=current_user.id,
        filename=file.filename,
        content_type=file.content_type,
        language=language,
    )

    # Validate file type
    allowed_types = {
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/mpga",
        "audio/m4a",
        "audio/wav",
        "audio/webm",
    }
    
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format: {file.content_type}. Supported: mp3, mp4, mpeg, mpga, m4a, wav, webm",
        )

    # Read audio file
    audio_content = await file.read()
    
    if len(audio_content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty",
        )

    # Check if STT is configured
    if not settings.STT_API_BASE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech-to-Text API is not configured. Please set STT_API_BASE environment variable.",
        )

    try:
        # Call STT API (OpenAI-compatible endpoint)
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {
                "file": (file.filename or "audio.mp3", audio_content, file.content_type or "audio/mpeg")
            }
            data = {
                "model": settings.STT_MODEL,
            }
            if language:
                data["language"] = language

            headers = {}
            if settings.STT_API_KEY:
                headers["Authorization"] = f"Bearer {settings.STT_API_KEY}"

            logger.info(
                "Calling STT API",
                api_base=settings.STT_API_BASE,
                model=settings.STT_MODEL,
                language=language,
            )

            response = await client.post(
                f"{settings.STT_API_BASE}/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
            )

            if response.status_code != 200:
                logger.error(
                    "STT API error",
                    status_code=response.status_code,
                    response=response.text,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Speech-to-Text API error: {response.text}",
                )

            result = response.json()
            transcribed_text = result.get("text", "")
            detected_language = result.get("language")

            logger.info(
                "Transcription completed",
                user_id=current_user.id,
                text_length=len(transcribed_text),
                detected_language=detected_language,
            )

            return VoiceTranscribeResponse(
                text=transcribed_text,
                language=detected_language,
            )

    except httpx.RequestError as e:
        logger.error("STT API request failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Speech-to-Text API: {str(e)}",
        )
    except Exception as e:
        logger.error("Transcription failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )


@router.post("/synthesize")
async def text_to_speech(
    request: VoiceTextToSpeechRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Convert text to speech using Text-to-Speech API.
    
    Returns audio in MP3 format.
    Uses OpenAI-compatible API endpoint configured in TTS_API_BASE.
    """
    logger.info(
        "Synthesizing speech",
        user_id=current_user.id,
        text_length=len(request.text),
        voice=request.voice,
        speed=request.speed,
    )

    # Check if TTS is configured
    if not settings.TTS_API_BASE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Text-to-Speech API is not configured. Please set TTS_API_BASE environment variable.",
        )

    try:
        # Call TTS API (OpenAI-compatible endpoint)
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {}
            if settings.TTS_API_KEY:
                headers["Authorization"] = f"Bearer {settings.TTS_API_KEY}"

            payload = {
                "model": settings.TTS_MODEL,
                "input": request.text,
                "voice": request.voice,
                "speed": request.speed,
            }

            logger.info(
                "Calling TTS API",
                api_base=settings.TTS_API_BASE,
                model=settings.TTS_MODEL,
                voice=request.voice,
                speed=request.speed,
            )

            response = await client.post(
                f"{settings.TTS_API_BASE}/audio/speech",
                json=payload,
                headers=headers,
            )

            if response.status_code != 200:
                logger.error(
                    "TTS API error",
                    status_code=response.status_code,
                    response=response.text,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Text-to-Speech API error: {response.text}",
                )

            audio_content = response.content

            logger.info(
                "Speech synthesis completed",
                user_id=current_user.id,
                audio_size=len(audio_content),
            )

            # Return audio as streaming response
            return StreamingResponse(
                io.BytesIO(audio_content),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3"
                },
            )

    except httpx.RequestError as e:
        logger.error("TTS API request failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Text-to-Speech API: {str(e)}",
        )
    except Exception as e:
        logger.error("Speech synthesis failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {str(e)}",
        )

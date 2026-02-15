"""API endpoints for translation operations."""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import structlog

from ..database import get_db, SessionLocal

logger = structlog.get_logger()

# Thread pool for running blocking translation jobs
_translation_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="translation_")
from ..services.auth import get_current_user
from ..services.translation_service import translation_service, SUPPORTED_LANGUAGES
from ..services.document_converter_service import document_converter_service
from ..services.document_translation_service import document_translation_service
from ..models import Agent, AgentType, TranslationJob, User, Conversation, Message
from ..schemas import (
    TranslationRequest,
    TranslationResponse,
    TranslationJobCreate,
    TranslationJobResponse,
    SupportedLanguagesResponse,
    TranslationChatRequest,
    ChatResponse,
    MessageResponse,
    TranslationMetadata,
)
from ..config import settings

router = APIRouter(prefix="/translation", tags=["translation"])


@router.get("/languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    """Get list of supported languages for translation."""
    return SupportedLanguagesResponse(languages=SUPPORTED_LANGUAGES)


@router.post("/{agent_id}/translate", response_model=TranslationResponse)
async def translate_text(
    agent_id: int,
    request: TranslationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Translate text using a translation agent."""
    # Get the agent
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.agent_type == AgentType.TRANSLATION,
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Translation agent not found")

    # Check access
    if agent.owner_id != current_user.id and not agent.is_public:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        result = await translation_service.translate_text(
            text=request.text,
            target_language=request.target_language,
            agent=agent,
            simplified=request.simplified,
            source_language=request.source_language,
        )

        return TranslationResponse(**result)

    except ValueError as e:
        # User-friendly error messages from translation service
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Translation error",
            agent_id=agent_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: Internal Server Error. Please try again or contact support if the issue persists."
        )


@router.post("/{agent_id}/chat", response_model=ChatResponse)
async def translation_chat(
    agent_id: int,
    request: TranslationChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Chat with a translation agent - translate text in conversation context."""
    # Get the agent
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.agent_type == AgentType.TRANSLATION,
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Translation agent not found")

    if agent.owner_id != current_user.id and not agent.is_public:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get or create conversation
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == current_user.id,
            Conversation.agent_id == agent_id,
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation(
            title=f"Translation: {request.message[:50]}...",
            user_id=current_user.id,
            agent_id=agent_id,
        )
        db.add(conversation)
        db.flush()

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )
    db.add(user_message)

    try:
        # Translate
        result = await translation_service.translate_text(
            text=request.message,
            target_language=request.target_language,
            agent=agent,
            simplified=request.simplified,
        )

        # Create translation metadata
        translation_meta = {
            "source_language": result["source_language"],
            "target_language": result["target_language"],
            "simplified": result["simplified"],
        }

        # Save assistant message
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result["translated_text"],
            translation_metadata=translation_meta,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)

        return ChatResponse(
            conversation_id=conversation.id,
            message=MessageResponse(
                id=assistant_message.id,
                conversation_id=conversation.id,
                role="assistant",
                content=assistant_message.content,
                translation_metadata=TranslationMetadata(**translation_meta),
                input_tokens=assistant_message.input_tokens,
                output_tokens=assistant_message.output_tokens,
                created_at=assistant_message.created_at,
                feedback=None,
            ),
            sources=[],
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


def run_translation_job_in_background(job_id: int):
    """
    Run the translation job in a separate thread with its own DB session.
    This prevents blocking the main FastAPI event loop.
    """
    db = SessionLocal()
    try:
        # Run the async function in a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_translation_job(job_id, db))
        finally:
            loop.close()
    finally:
        db.close()


async def process_translation_job(job_id: int, db: Session):
    """
    Background task to process a file translation job.

    Universal Pivot Pipeline (Format-Preserving):
    1. Convert to editable format if needed (PDF → DOCX)
    2. Extract text segments to JSON structure
    3. Translate segments via LLM
    4. Re-inject translated text with font scaling
    5. Convert back to original format

    This preserves document layout, images, tables, and styling.
    """
    import time
    start_time = time.time()

    logger.info(f"[Job {job_id}] Starting format-preserving translation job")

    job = db.query(TranslationJob).filter(TranslationJob.id == job_id).first()
    if not job:
        logger.error(f"[Job {job_id}] Job not found in database")
        return

    agent = db.query(Agent).filter(Agent.id == job.agent_id).first()
    if not agent:
        job.status = "failed"
        job.error_message = "Agent not found"
        job.processing_time_seconds = time.time() - start_time
        db.commit()
        logger.error(f"[Job {job_id}] Agent not found")
        return

    # Get job directory (created during upload)
    job_dir = Path(job.job_directory) if job.job_directory else Path(settings.UPLOAD_DIR) / "jobs" / "translation" / str(job.user_id) / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_ext = job.source_file_type.lower()
        source_stem = Path(job.source_filename).stem
        mode_suffix = "_simplified" if job.simplified_mode else ""

        # Determine if we can use format-preserving translation
        format_preserving_types = {"pdf", "docx", "doc", "pptx", "ppt"}
        use_format_preserving = source_ext in format_preserving_types

        if use_format_preserving:
            # ================================================================
            # FORMAT-PRESERVING WORKFLOW (PDF, DOCX, PPTX)
            # ================================================================

            # Step 1: Converting to editable format (if PDF)
            if source_ext == "pdf":
                logger.info(f"[Job {job_id}] Phase 1/4: Converting PDF to DOCX")
                job.status = "extracting"
                job.progress = 10
                job.error_message = None
                db.commit()

                # Convert PDF to DOCX for editing (save in job directory)
                docx_filename = f"converted_{source_stem}.docx"
                docx_path = str(job_dir / docx_filename)
                await document_translation_service.pdf_to_docx(
                    job.source_file_path, output_path=docx_path
                )
                work_path = docx_path
                work_format = "docx"
                job.progress = 25
                db.commit()
            else:
                work_path = job.source_file_path
                work_format = source_ext
                job.status = "extracting"
                job.progress = 15
                db.commit()

            # Step 2: Extract document structure
            logger.info(f"[Job {job_id}] Phase 2/4: Extracting document structure")

            if work_format in ("docx", "doc"):
                structure = await document_translation_service.extract_docx_structure(work_path)
            elif work_format in ("pptx", "ppt"):
                structure = await document_translation_service.extract_pptx_structure(work_path)
            else:
                raise ValueError(f"Unsupported format for structure extraction: {work_format}")

            segments = structure.get_translation_payload()
            if not segments:
                raise ValueError("No translatable content found in document")

            job.extracted_markdown = structure.to_json()  # Store structure as JSON
            job.progress = 35
            db.commit()
            logger.info(f"[Job {job_id}] Extracted {len(segments)} text segments")

            # Step 3: Translate segments
            logger.info(f"[Job {job_id}] Phase 3/4: Translating {len(segments)} segments to {job.target_language}")
            job.status = "translating"
            job.progress = 40
            db.commit()

            # Create progress callback to update database during translation
            async def translation_progress(completed: int, total: int):
                # Progress from 40% to 70% during translation (30% range)
                progress = 40 + int((completed / total) * 30)
                job.progress = progress
                db.commit()
                logger.debug(f"[Job {job_id}] Translation progress: {completed}/{total} batches ({progress}%)")

            translations = await document_translation_service.translate_segments(
                segments=segments,
                target_language=job.target_language,
                simplified=job.simplified_mode,
                llm_model=agent.llm_model,
                agent=agent,
                progress_callback=translation_progress,
            )

            structure.apply_translations(translations)
            job.translated_markdown = structure.to_json()  # Store translated structure
            job.progress = 70
            db.commit()
            logger.info(f"[Job {job_id}] Translated {len(translations)}/{len(segments)} segments")

            # Step 4: Re-inject and generate output
            logger.info(f"[Job {job_id}] Phase 4/4: Re-injecting translations with font scaling")
            job.status = "generating"
            job.progress = 80
            db.commit()

            # Determine output format (same as input)
            if source_ext == "pdf":
                # PDF: output as DOCX (PDF conversion requires LibreOffice)
                output_filename = f"translated_{source_stem}_{job.target_language}{mode_suffix}.docx"
            elif source_ext in ("docx", "doc"):
                output_filename = f"translated_{source_stem}_{job.target_language}{mode_suffix}.docx"
            elif source_ext in ("pptx", "ppt"):
                output_filename = f"translated_{source_stem}_{job.target_language}{mode_suffix}.pptx"
            else:
                output_filename = f"translated_{source_stem}_{job.target_language}{mode_suffix}.{source_ext}"

            # Store output in job directory
            output_path = str(job_dir / output_filename)

            # Re-inject translations
            if work_format in ("docx", "doc"):
                await document_translation_service.inject_docx_translations(
                    work_path, structure, output_path, scale_fonts=True
                )
            elif work_format in ("pptx", "ppt"):
                await document_translation_service.inject_pptx_translations(
                    work_path, structure, output_path, scale_fonts=True
                )

            job.output_filename = output_filename
            job.output_file_path = output_path

        else:
            # ================================================================
            # LEGACY WORKFLOW (TXT, MD - simple text files)
            # ================================================================

            logger.info(f"[Job {job_id}] Phase 1/4: Extracting content from {job.source_filename}")
            job.status = "extracting"
            job.progress = 10
            job.error_message = None
            db.commit()

            # Extract text content
            extracted_text = await document_converter_service.extract_text_from_file(
                file_path=job.source_file_path,
                file_type=job.source_file_type,
                use_llm=False,
            )

            if not extracted_text or len(extracted_text.strip()) == 0:
                raise ValueError("No content could be extracted from the file")

            job.extracted_markdown = extracted_text
            job.progress = 30
            db.commit()

            # Translate
            logger.info(f"[Job {job_id}] Phase 2/4: Translating to {job.target_language}")
            job.status = "translating"
            job.progress = 40
            db.commit()

            translation_result = await translation_service.translate_markdown(
                markdown=job.extracted_markdown,
                target_language=job.target_language,
                agent=agent,
                simplified=job.simplified_mode,
            )
            job.translated_markdown = translation_result["translated_markdown"]
            job.input_tokens = translation_result.get("input_tokens")
            job.output_tokens = translation_result.get("output_tokens")
            job.progress = 70
            db.commit()

            # Generate output (same format as input for text files)
            logger.info(f"[Job {job_id}] Phase 3/4: Generating output")
            job.status = "generating"
            job.progress = 80
            db.commit()

            output_filename = f"translated_{source_stem}_{job.target_language}{mode_suffix}.{source_ext}"
            # Store output in job directory
            output_path = str(job_dir / output_filename)

            # Write translated text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(job.translated_markdown)

            job.output_filename = output_filename
            job.output_file_path = output_path

        # Mark as completed
        job.status = "completed"
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.processing_time_seconds = round(time.time() - start_time, 2)
        db.commit()

        logger.info(f"[Job {job_id}] Job completed successfully - {job.output_filename} in {job.processing_time_seconds}s")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Job {job_id}] FAILED: {error_msg}")
        logger.error(f"[Job {job_id}] Traceback: {traceback.format_exc()}")
        job.status = "failed"
        job.error_message = error_msg[:500]
        job.processing_time_seconds = round(time.time() - start_time, 2)
        db.commit()
    finally:
        # Clean up temp files
        try:
            document_translation_service.cleanup()
        except Exception:
            pass


@router.post("/{agent_id}/translate-file", response_model=TranslationJobResponse)
async def translate_file(
    agent_id: int,
    file: UploadFile = File(...),
    target_language: str = "en",
    simplified: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a file for translation. Returns a job ID to track progress."""
    # Get the agent
    agent = db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.agent_type == AgentType.TRANSLATION,
    ).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Translation agent not found")

    if agent.owner_id != current_user.id and not agent.is_public:
        raise HTTPException(status_code=403, detail="Access denied")

    # Validate file type
    file_ext = Path(file.filename).suffix.lower().lstrip(".")
    supported_types = ["pdf", "txt", "md", "docx", "doc", "pptx", "ppt"]
    if file_ext not in supported_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(supported_types)}"
        )

    # Create job-specific directory with unique ID (prefixed with timestamp for easy sorting)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    job_uuid = f"{timestamp}_{str(uuid.uuid4())[:8]}"
    job_dir = Path(settings.UPLOAD_DIR) / "jobs" / "translation" / str(current_user.id) / job_uuid
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file in job directory
    source_file_path = job_dir / f"source_{file.filename}"

    with open(source_file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create translation job with job directory
    job = TranslationJob(
        agent_id=agent_id,
        user_id=current_user.id,
        job_directory=str(job_dir),
        source_filename=file.filename,
        source_file_path=str(source_file_path),
        source_file_type=file_ext,
        target_language=target_language,
        simplified_mode=simplified,
        status="pending",
        progress=0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start background processing in a separate thread to not block the event loop
    _translation_executor.submit(run_translation_job_in_background, job.id)
    logger.info(f"[Job {job.id}] Submitted to translation thread pool")

    return TranslationJobResponse(
        id=job.id,
        agent_id=job.agent_id,
        source_filename=job.source_filename,
        target_language=job.target_language,
        simplified_mode=job.simplified_mode,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        output_filename=job.output_filename,
        job_directory=job.job_directory,
        input_tokens=job.input_tokens,
        output_tokens=job.output_tokens,
        processing_time_seconds=job.processing_time_seconds,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs/{job_id}", response_model=TranslationJobResponse)
async def get_translation_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the status of a translation job."""
    job = db.query(TranslationJob).filter(
        TranslationJob.id == job_id,
        TranslationJob.user_id == current_user.id,
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Translation job not found")

    return TranslationJobResponse(
        id=job.id,
        agent_id=job.agent_id,
        source_filename=job.source_filename,
        target_language=job.target_language,
        simplified_mode=job.simplified_mode,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        output_filename=job.output_filename,
        job_directory=job.job_directory,
        input_tokens=job.input_tokens,
        output_tokens=job.output_tokens,
        processing_time_seconds=job.processing_time_seconds,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs/{job_id}/download")
async def download_translated_file(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the translated PDF file."""
    job = db.query(TranslationJob).filter(
        TranslationJob.id == job_id,
        TranslationJob.user_id == current_user.id,
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Translation job not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Translation not completed yet")

    if not job.output_file_path or not os.path.exists(job.output_file_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    # Determine media type based on file extension
    ext = Path(job.output_filename).suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".ppt": "application/vnd.ms-powerpoint",
        ".html": "text/html",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=job.output_file_path,
        filename=job.output_filename,
        media_type=media_type,
    )


@router.get("/jobs", response_model=list[TranslationJobResponse])
async def list_translation_jobs(
    agent_id: int | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List translation jobs for the current user."""
    query = db.query(TranslationJob).filter(TranslationJob.user_id == current_user.id)

    if agent_id:
        query = query.filter(TranslationJob.agent_id == agent_id)
    if status:
        query = query.filter(TranslationJob.status == status)

    jobs = query.order_by(TranslationJob.created_at.desc()).limit(50).all()

    return [
        TranslationJobResponse(
            id=job.id,
            agent_id=job.agent_id,
            source_filename=job.source_filename,
            target_language=job.target_language,
            simplified_mode=job.simplified_mode,
            status=job.status,
            progress=job.progress,
            error_message=job.error_message,
            output_filename=job.output_filename,
            job_directory=job.job_directory,
            input_tokens=job.input_tokens,
            output_tokens=job.output_tokens,
            processing_time_seconds=job.processing_time_seconds,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


"""Agent API endpoints - supports both RAG and Translation agents."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
import structlog

from ..database import get_db
from ..models import User, Agent, DataDomain, AgentType
from ..schemas import (
    RAGAgentCreate,
    TranslationAgentCreate,
    VideoTranscriptionAgentCreate,
    AvatarAgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentListResponse,
    AgentType as SchemaAgentType,
)
from ..services.auth import get_current_active_user
from ..config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.get("/", response_model=list[AgentListResponse])
async def list_agents(
    agent_type: SchemaAgentType | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all agents for the current user (including public agents)."""
    from ..services.model_service import model_service

    query = db.query(Agent).filter(
        (Agent.owner_id == current_user.id) | (Agent.is_public == True)
    )

    if agent_type:
        query = query.filter(Agent.agent_type == agent_type)

    agents = query.all()
    logger.info(
        "Agents list",
        agent_type_filter=agent_type,
        count=len(agents),
        agent_types=[a.agent_type for a in agents],
    )

    # Check model availability for each agent
    llm_models = await model_service.get_llm_models(refresh=False)
    available_model_ids = {m.id for m in llm_models}

    result = []
    for agent in agents:
        agent_dict = {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "agent_type": agent.agent_type,
            "llm_model": agent.llm_model,
            "data_domain_ids": [d.id for d in agent.data_domains],
            "target_language": agent.target_language,
            "owner_id": agent.owner_id,
            "is_public": agent.is_public,
            "model_available": agent.llm_model is not None
            and agent.llm_model in available_model_ids,
            "created_at": agent.created_at,
        }
        result.append(AgentListResponse(**agent_dict))

    return result


@router.post("/rag", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_rag_agent(
    agent_data: RAGAgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new RAG agent."""
    domain_ids = agent_data.data_domain_ids or []
    if not domain_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one data domain is required for a RAG agent",
        )
    domains = (
        db.query(DataDomain)
        .filter(
            DataDomain.id.in_(domain_ids),
            DataDomain.owner_id == current_user.id,
        )
        .all()
    )
    if len(domains) != len(domain_ids):
        found = {d.id for d in domains}
        missing = set(domain_ids) - found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data domain(s) not found or access denied: {sorted(missing)}",
        )

    agent = Agent(
        name=agent_data.name,
        description=agent_data.description,
        agent_type=AgentType.RAG,
        llm_model=agent_data.llm_model,
        system_prompt=agent_data.system_prompt,
        temperature=agent_data.temperature,
        top_p=agent_data.top_p,
        top_k=agent_data.top_k,
        max_tokens=agent_data.max_tokens,
        retrieval_k=agent_data.retrieval_k,
        test_questions=agent_data.test_questions,
        is_public=agent_data.is_public,
        owner_id=current_user.id,
    )
    db.add(agent)
    db.flush()
    agent.data_domains = domains
    db.commit()
    db.refresh(agent)

    logger.info("Created RAG agent", agent_id=agent.id, name=agent.name)
    return agent


@router.post(
    "/translation", response_model=AgentResponse, status_code=status.HTTP_201_CREATED
)
async def create_translation_agent(
    agent_data: TranslationAgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new Translation agent."""
    agent = Agent(
        name=agent_data.name,
        description=agent_data.description,
        agent_type=AgentType.TRANSLATION,
        llm_model=agent_data.llm_model,
        system_prompt=agent_data.system_prompt,
        temperature=agent_data.temperature,
        max_tokens=agent_data.max_tokens,
        source_language=agent_data.source_language,
        target_language=agent_data.target_language,
        supported_languages=agent_data.supported_languages,
        is_public=agent_data.is_public,
        owner_id=current_user.id,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    logger.info("Created Translation agent", agent_id=agent.id, name=agent.name)
    return agent


@router.post(
    "/video-transcription",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_video_transcription_agent(
    agent_data: VideoTranscriptionAgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new Video Transcription agent. Requires WHISPER_API_BASE to be configured."""
    from ..config import settings

    # Check if Whisper API is configured
    if not settings.WHISPER_API_BASE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video transcription is not available. WHISPER_API_BASE must be configured in environment variables.",
        )
    # Create agent
    # whisper_model_size is optional - only used for custom Whisper services, not OpenAI
    agent = Agent(
        name=agent_data.name,
        description=agent_data.description,
        agent_type=AgentType.VIDEO_TRANSCRIPTION,
        llm_model=None,  # Not needed for transcription
        temperature=None,
        max_tokens=None,
        system_prompt=None,
        whisper_model_size=agent_data.whisper_model_size
        or "base",  # Default to base if not provided
        owner_id=current_user.id,
        is_public=agent_data.is_public,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    logger.info(
        "Created Video Transcription agent",
        agent_id=agent.id,
        name=agent.name,
        whisper_model_size=agent_data.whisper_model_size,
    )
    return agent


@router.get("/video-transcription/config")
async def get_video_transcription_config():
    """Check if video transcription is available (Whisper API configured)."""
    available = bool(settings.WHISPER_API_BASE)
    logger.info(
        "Video transcription config requested",
        available=available,
        whisper_configured=bool(settings.WHISPER_API_BASE),
    )
    return {
        "available": available,
        "whisper_api_base": settings.WHISPER_API_BASE
        if settings.WHISPER_API_BASE
        else None,
    }


@router.post(
    "/avatar",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_avatar_agent(
    agent_data: AvatarAgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new Avatar agent backed by a LiveKit room and SoulX-LiveTalk."""
    agent = Agent(
        name=agent_data.name,
        description=agent_data.description,
        agent_type=AgentType.AVATAR,
        llm_model=agent_data.llm_model,
        system_prompt=agent_data.system_prompt,
        temperature=agent_data.temperature,
        max_tokens=agent_data.max_tokens,
        retrieval_k=agent_data.retrieval_k,
        avatar_config=agent_data.avatar_config,
        is_public=agent_data.is_public,
        owner_id=current_user.id,
    )
    db.add(agent)
    db.flush()

    # Optionally attach RAG data domains
    if agent_data.data_domain_ids:
        domains = (
            db.query(DataDomain)
            .filter(
                DataDomain.id.in_(agent_data.data_domain_ids),
                DataDomain.owner_id == current_user.id,
            )
            .all()
        )
        agent.data_domains = domains

    db.commit()
    db.refresh(agent)

    logger.info("Created Avatar agent", agent_id=agent.id, name=agent.name)
    return agent


# Keep the old POST endpoint for backwards compatibility (defaults to RAG)
@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: RAGAgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new RAG agent (legacy endpoint, use /agents/rag instead)."""
    return await create_rag_agent(agent_data, db, current_user)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific agent."""
    from ..services.model_service import model_service

    agent = (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            (Agent.owner_id == current_user.id) | (Agent.is_public == True),
        )
        .first()
    )

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    # Check model availability
    llm_models = await model_service.get_llm_models(refresh=False)
    available_model_ids = {m.id for m in llm_models}
    model_available = agent.llm_model in available_model_ids

    # Create response with model availability
    agent_dict = {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "agent_type": agent.agent_type,
        "llm_model": agent.llm_model,
        "system_prompt": agent.system_prompt,
        "temperature": agent.temperature,
        "top_p": agent.top_p,
        "top_k": agent.top_k,
        "max_tokens": agent.max_tokens,
        "data_domain_ids": [d.id for d in agent.data_domains],
        "retrieval_k": agent.retrieval_k,
        "source_language": agent.source_language,
        "target_language": agent.target_language,
        "supported_languages": agent.supported_languages,
        "avatar_config": agent.avatar_config,
        "owner_id": agent.owner_id,
        "is_public": agent.is_public,
        "test_questions": agent.test_questions,
        "model_available": model_available,
        "created_at": agent.created_at,
        "updated_at": agent.updated_at,
    }

    return AgentResponse(**agent_dict)


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    agent_data: AgentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update an agent (only owner can update)."""
    agent = (
        db.query(Agent)
        .filter(Agent.id == agent_id, Agent.owner_id == current_user.id)
        .first()
    )

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or access denied",
        )

    # If changing data domains (only for RAG agents), verify they exist
    if agent_data.data_domain_ids is not None and agent.agent_type == AgentType.RAG:
        domain_ids = agent_data.data_domain_ids
        domains = (
            db.query(DataDomain)
            .filter(
                DataDomain.id.in_(domain_ids),
                DataDomain.owner_id == current_user.id,
            )
            .all()
        )
        if len(domains) != len(domain_ids):
            found = {d.id for d in domains}
            missing = set(domain_ids) - found
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data domain(s) not found or access denied: {sorted(missing)}",
            )
        agent.data_domains = domains

    # Update other fields (exclude data_domain_ids - we handled it above)
    update_data = agent_data.model_dump(exclude_unset=True, exclude={"data_domain_ids"})
    for field, value in update_data.items():
        setattr(agent, field, value)

    db.commit()
    db.refresh(agent)

    logger.info("Updated agent", agent_id=agent_id, agent_type=agent.agent_type)
    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete an agent (only owner can delete). Deletes all related conversations, translation jobs, but NOT data domains."""
    from ..models import Conversation, Message, TranslationJob

    # Find the agent (check ownership separately for better error messages)
    agent = db.query(Agent).filter(Agent.id == agent_id).first()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    # Check ownership (allow superusers to delete any agent)
    if agent.owner_id != current_user.id and not current_user.is_superuser:
        logger.warning(
            "Attempt to delete agent by non-owner",
            agent_id=agent_id,
            agent_name=agent.name,
            agent_owner_id=agent.owner_id,
            current_user_id=current_user.id,
            current_user_email=current_user.email,
            is_superuser=current_user.is_superuser,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete agents that you own",
        )

    # Count related records before deletion for logging
    conversation_count = (
        db.query(Conversation).filter(Conversation.agent_id == agent_id).count()
    )
    translation_job_count = (
        db.query(TranslationJob).filter(TranslationJob.agent_id == agent_id).count()
    )

    # Explicitly delete all translation jobs for this agent first
    translation_jobs = (
        db.query(TranslationJob).filter(TranslationJob.agent_id == agent_id).all()
    )
    for job in translation_jobs:
        db.delete(job)

    # Explicitly delete all conversations for this agent (and their messages via cascade)
    # This ensures conversations are deleted even if cascade doesn't work as expected
    conversations = (
        db.query(Conversation).filter(Conversation.agent_id == agent_id).all()
    )
    for conversation in conversations:
        # Messages will be deleted via cascade from Conversation model
        db.delete(conversation)

    # Delete the agent (data domains are NOT deleted - they're independent)
    db.delete(agent)
    db.commit()

    logger.info(
        "Deleted agent and related data",
        agent_id=agent_id,
        agent_name=agent.name,
        conversations_deleted=conversation_count,
        translation_jobs_deleted=translation_job_count,
    )


@router.post("/{agent_id}/test")
async def test_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Run test questions against a RAG agent."""
    agent = (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            (Agent.owner_id == current_user.id) | (Agent.is_public == True),
        )
        .first()
    )

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    if agent.agent_type != AgentType.RAG:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Test is only available for RAG agents",
        )

    if not agent.test_questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent has no test questions configured",
        )

    domains = list(agent.data_domains)
    if not domains:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent has no data domains configured",
        )

    from ..services.rag_service import rag_service

    results = []
    for test in agent.test_questions:
        question = test.get("question", "")
        reference_answer = test.get("reference_answer", "")

        try:
            response = rag_service.query(question, agent, domains)
            results.append(
                {
                    "question": question,
                    "reference_answer": reference_answer,
                    "agent_response": response["response"],
                    "sources": response["sources"],
                }
            )
        except Exception as e:
            results.append(
                {
                    "question": question,
                    "reference_answer": reference_answer,
                    "error": str(e),
                }
            )

    return {"results": results}

"""Conversation and Chat API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, Agent, DataDomain, Conversation, Message, Feedback
from ..schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationListResponse,
    FeedbackResponse,
    MessageResponse,
    ChatRequest,
    ChatResponse,
    FeedbackCreate,
    FeedbackUpdate,
    SourceReference,
)
from ..services.auth import get_current_active_user
from ..services.rag_service import rag_service
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/conversations", tags=["Conversations"])


def _conversation_to_response(
    conversation: Conversation, include_messages: bool = False
) -> ConversationResponse:
    """Convert Conversation model to ConversationResponse schema."""
    from ..schemas import LifecycleState, AccessPolicy, MessageResponse

    messages = []
    if include_messages and conversation.messages:
        # Convert messages to MessageResponse format
        for msg in conversation.messages:
            messages.append(
                MessageResponse(
                    id=str(msg.id),
                    session_id=str(conversation.id),
                    role=msg.role,
                    content=msg.content,
                    parent_message_id=str(msg.parent_message_id)
                    if msg.parent_message_id
                    else None,
                    version_label=msg.version_label,
                    is_active_path=msg.is_active_path
                    if msg.is_active_path is not None
                    else True,
                    sources=msg.sources,  # Already in correct format if present
                    input_tokens=msg.input_tokens,
                    output_tokens=msg.output_tokens,
                    created_at=msg.created_at,
                    feedback=FeedbackResponse(
                        id=msg.feedback.id,
                        message_id=str(msg.feedback.message_id),
                        rating_type=msg.feedback.rating_type,
                        score=msg.feedback.score,
                        thumbs_up=msg.feedback.thumbs_up,
                        tags=msg.feedback.tags or [],
                        comment=msg.feedback.comment,
                        user_id=msg.feedback.user_id,
                        created_at=msg.feedback.created_at,
                        updated_at=msg.feedback.updated_at,
                    ) if msg.feedback else None,
                    child_messages=[],
                    citations=[],
                    translation_metadata=None,
                )
            )

    # Get title from metadata.topic
    # Safely access metadata column (avoid conflict with SQLAlchemy Base.metadata)
    metadata = getattr(conversation, "conversation_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}

    title = metadata.get("topic") if metadata else None

    return ConversationResponse(
        id=str(conversation.id),
        title=title,  # Title comes from metadata.topic
        metadata=metadata if metadata else None,
        agent_id=conversation.agent_id,
        owner_id=conversation.user_id,  # Map user_id to owner_id
        tenant_id=None,
        access_policy=AccessPolicy.PRIVATE,
        allowed_users=[],
        lifecycle_state=LifecycleState.ACTIVE,
        deleted_at=None,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=messages,
    )


@router.get("/", response_model=list[ConversationListResponse])
async def list_conversations(
    agent_id: int | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all conversations for the current user."""
    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)

    if agent_id:
        query = query.filter(Conversation.agent_id == agent_id)

    conversations = query.order_by(Conversation.updated_at.desc()).all()

    logger.info(
        "Listing conversations",
        user_id=current_user.id,
        agent_id=agent_id,
        total_conversations=len(conversations),
    )

    result = []
    for conv in conversations:
        msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        agent = db.query(Agent).filter(Agent.id == conv.agent_id).first()
        # Get title from metadata.topic (use conversation_metadata attribute name)
        metadata = getattr(conv, "conversation_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        title = metadata.get("topic") if metadata else None
        # Default to "Untitled" if no topic is set
        if not title:
            title = "Untitled"
        result.append(
            ConversationListResponse(
                id=conv.id,
                title=title,
                agent_id=conv.agent_id,
                agent_name=agent.name if agent else "Unknown",
                agent_type=agent.agent_type if agent else None,
                owner_id=getattr(conv, "owner_id", None)
                or getattr(
                    conv, "user_id", None
                ),  # Support both old and new field names
                access_policy=getattr(conv, "access_policy", None),
                lifecycle_state=getattr(conv, "lifecycle_state", None),
                message_count=msg_count,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            )
        )
        logger.debug(
            "Added conversation to result",
            conversation_id=conv.id,
            title=title,
            agent_id=conv.agent_id,
            message_count=msg_count,
        )

    logger.info(
        "Returning conversations list",
        count=len(result),
        agent_id=agent_id,
        user_id=current_user.id,
        sample_titles=[r.title for r in result[:5]] if result else [],
    )
    return result


@router.post(
    "/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    conv_data: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new conversation."""
    # Verify agent exists and is accessible
    agent = (
        db.query(Agent)
        .filter(
            Agent.id == conv_data.agent_id,
            (Agent.owner_id == current_user.id) | (Agent.is_public == True),
        )
        .first()
    )

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or access denied",
        )

    # Determine title from metadata.topic or title field (for backward compatibility), with fallback
    default_title = f"Chat with {agent.name}"
    metadata = conv_data.metadata or {}

    # Ensure metadata is a dict
    if not isinstance(metadata, dict):
        metadata = {}

    # If metadata.topic is provided, use it
    # If title is provided (backward compatibility), set metadata.topic
    if conv_data.title and not metadata.get("topic"):
        metadata["topic"] = conv_data.title
    # If neither provided, set default
    elif not metadata.get("topic"):
        metadata["topic"] = default_title

    # Create conversation with metadata.topic
    conversation = Conversation(
        conversation_metadata=metadata,  # Use conversation_metadata attribute name
        user_id=current_user.id,
        agent_id=conv_data.agent_id,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    logger.info(
        "Created conversation", conversation_id=conversation.id, agent_id=agent.id
    )

    # Return response matching ConversationResponse schema
    return _conversation_to_response(conversation, include_messages=False)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific conversation with all messages."""
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id, Conversation.user_id == current_user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )

    return _conversation_to_response(conversation, include_messages=True)


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    conv_data: ConversationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update a conversation (e.g., title)."""
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id, Conversation.user_id == current_user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )

    # Update conversation - sync title and metadata.topic
    # Safely access metadata column (avoid conflict with SQLAlchemy Base.metadata)
    current_metadata = getattr(conversation, "conversation_metadata", None)
    if not isinstance(current_metadata, dict):
        current_metadata = {}

    if conv_data.title is not None:
        # Update metadata.topic from title field (backward compatibility)
        current_metadata["topic"] = conv_data.title
        conversation.conversation_metadata = current_metadata

    if conv_data.metadata is not None:
        conversation.conversation_metadata = conv_data.metadata

    db.commit()
    db.refresh(conversation)

    return _conversation_to_response(conversation, include_messages=False)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a conversation."""
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id, Conversation.user_id == current_user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )

    db.delete(conversation)
    db.commit()

    logger.info("Deleted conversation", conversation_id=conversation_id)


@router.post("/{conversation_id}/chat", response_model=ChatResponse)
async def chat(
    conversation_id: int,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Send a message in a conversation and get a response."""
    # Get conversation
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id, Conversation.user_id == current_user.id
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )

    # Get agent
    agent = db.query(Agent).filter(Agent.id == conversation.agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    domains = list(agent.data_domains)
    if not domains:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent has no data domains configured",
        )

    # Save user message
    user_message = Message(
        conversation_id=conversation_id, role="user", content=chat_request.message
    )
    db.add(user_message)
    db.commit()

    # Generate title from first question BEFORE getting the answer
    # Check metadata.topic for default title pattern
    # Safely access metadata column (avoid conflict with SQLAlchemy Base.metadata)
    metadata = getattr(conversation, "conversation_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}

    current_title = metadata.get("topic")

    # Generate title if topic is missing, empty, "Untitled", or starts with "Chat with"
    if (
        not current_title
        or current_title == "Untitled"
        or current_title.startswith("Chat with")
    ):
        logger.info(
            "=== TITLE GENERATION START ===",
            conversation_id=conversation_id,
            query_message=chat_request.message,
            current_title=current_title,
        )
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from ..config import settings
            from ..services.model_service import model_service
            import asyncio

            # Use agent's model for title generation (no restrictions on model name)
            title_model = agent.llm_model
            logger.info(
                "Using agent's model for title generation",
                model=title_model,
                conversation_id=conversation_id,
            )

            # Generate title using the agent's model
            if title_model:
                # Generate a concise title from the user's message (similar to working curl example)
                title_prompt = (
                    f"what is a 3 word title for this question: {chat_request.message}"
                )

                logger.info(
                    "=== TITLE GENERATION PROMPT ===",
                    conversation_id=conversation_id,
                    model=title_model,
                    title_prompt=title_prompt,
                )

                llm = ChatOpenAI(
                    model=title_model,
                    temperature=0.3,  # Lower temperature for more consistent titles
                    max_tokens=1024,  # Increased from 10 to allow model to respond properly
                    api_key=settings.LLM_API_KEY,
                    base_url=settings.LLM_API_BASE,
                )

                logger.info(
                    "Calling LLM for title generation",
                    model=title_model,
                    conversation_id=conversation_id,
                )
                response = llm.invoke([HumanMessage(content=title_prompt)])
                # Try different ways to get the content
                raw_title = None
                if hasattr(response, "content"):
                    raw_title = response.content.strip() if response.content else None
                elif hasattr(response, "text"):
                    raw_title = response.text.strip() if response.text else None
                elif isinstance(response, str):
                    raw_title = response.strip()

                logger.info(
                    "=== LLM TITLE RESPONSE ===",
                    conversation_id=conversation_id,
                    raw_response=raw_title,
                    response_type=type(response).__name__,
                    response_attrs=dir(response)
                    if hasattr(response, "__dict__")
                    else None,
                )

                # Process the title
                if raw_title and len(raw_title.strip()) > 0:
                    # Remove quotes if present and clean up
                    generated_title = raw_title.strip('"').strip("'").strip()
                    # Remove "Title:" prefix if present
                    if generated_title.lower().startswith("title:"):
                        generated_title = generated_title[6:].strip()

                    # Remove markdown formatting (bold, italic, code, links, etc.)
                    import re

                    # Remove markdown bold/italic: **text**, *text*, __text__, _text_
                    generated_title = re.sub(r"\*\*([^*]+)\*\*", r"\1", generated_title)
                    generated_title = re.sub(r"\*([^*]+)\*", r"\1", generated_title)
                    generated_title = re.sub(r"__([^_]+)__", r"\1", generated_title)
                    generated_title = re.sub(r"_([^_]+)_", r"\1", generated_title)
                    # Remove markdown code: `code`
                    generated_title = re.sub(r"`([^`]+)`", r"\1", generated_title)
                    # Remove markdown links: [text](url)
                    generated_title = re.sub(
                        r"\[([^\]]+)\]\([^\)]+\)", r"\1", generated_title
                    )
                    # Remove any remaining markdown special characters
                    generated_title = (
                        generated_title.replace("#", "")
                        .replace("##", "")
                        .replace("###", "")
                    )
                    # Clean up extra spaces
                    generated_title = " ".join(generated_title.split())

                    # Truncate to 60 chars for display
                    if generated_title and len(generated_title) > 0:
                        final_title = generated_title[:60]
                        # Update metadata.topic
                        if not isinstance(metadata, dict):
                            metadata = {}
                        metadata["topic"] = final_title
                        conversation.conversation_metadata = metadata
                        from sqlalchemy.orm.attributes import flag_modified

                        flag_modified(conversation, "conversation_metadata")
                        logger.info(
                            "=== TITLE GENERATED ===",
                            conversation_id=conversation_id,
                            generated_title=final_title,
                            metadata=metadata,
                        )
                    else:
                        if not isinstance(metadata, dict):
                            metadata = {}
                        metadata["topic"] = "Untitled"
                        conversation.conversation_metadata = metadata
                        from sqlalchemy.orm.attributes import flag_modified

                        flag_modified(conversation, "conversation_metadata")
                        logger.info(
                            "Title was empty after processing, using 'Untitled'",
                            conversation_id=conversation_id,
                        )
                else:
                    if not isinstance(metadata, dict):
                        metadata = {}
                    metadata["topic"] = "Untitled"
                    conversation.conversation_metadata = metadata
                    from sqlalchemy.orm.attributes import flag_modified

                    flag_modified(conversation, "conversation_metadata")
                    logger.info(
                        "LLM returned None/empty, using 'Untitled'",
                        conversation_id=conversation_id,
                    )

                # Commit title update
                db.commit()
                db.refresh(conversation)
        except Exception as e:
            logger.error(
                "Failed to generate conversation title",
                error=str(e),
                conversation_id=conversation_id,
                exc_info=True,
            )
            # Set to "Untitled" if generation fails
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["topic"] = "Untitled"
            conversation.conversation_metadata = metadata
            from sqlalchemy.orm.attributes import flag_modified

            flag_modified(conversation, "conversation_metadata")
            db.commit()
            db.refresh(conversation)

    # Generate response using RAG
    logger.info(
        "=== GENERATING RAG RESPONSE ===",
        conversation_id=conversation_id,
        query_message=chat_request.message,
        agent_id=agent.id,
        agent_name=agent.name,
    )
    try:
        rag_response = rag_service.query(chat_request.message, agent, domains)
        logger.info(
            "=== RAG RESPONSE RECEIVED ===",
            conversation_id=conversation_id,
            response_length=len(rag_response.get("response", "")),
            response_preview=rag_response.get("response", "")[:200] + "..."
            if len(rag_response.get("response", "")) > 200
            else rag_response.get("response", ""),
            input_tokens=rag_response.get("input_tokens"),
            output_tokens=rag_response.get("output_tokens"),
            sources_count=len(rag_response.get("sources", [])),
        )
    except Exception as e:
        logger.error("RAG query failed", error=str(e), conversation_id=conversation_id)
        # Don't expose internal error details to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response. Please try again or contact support.",
        )

    # Save assistant message
    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=rag_response["response"],
        input_tokens=rag_response.get("input_tokens"),
        output_tokens=rag_response.get("output_tokens"),
    )
    # Set sources after creation (using setattr to avoid constructor issues)
    try:
        setattr(assistant_message, "sources", rag_response["sources"])
    except AttributeError:
        pass  # Column doesn't exist yet, skip
    db.add(assistant_message)

    # Update conversation timestamp
    from datetime import datetime, timezone

    conversation.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(assistant_message)
    db.refresh(conversation)

    # Log final conversation state after commit
    metadata = getattr(conversation, "conversation_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    title = metadata.get("topic")
    logger.info(
        "Conversation state after commit",
        conversation_id=conversation_id,
        title=title,
        title_length=len(title) if title else 0,
        updated_at=conversation.updated_at,
    )

    logger.info(
        "Chat completed",
        conversation_id=conversation_id,
        input_tokens=rag_response.get("input_tokens"),
        output_tokens=rag_response.get("output_tokens"),
    )

    # Build response (document_id can be None when Qdrant payload has no document_id, e.g. from ingestion worker)
    def _doc_id(s: dict) -> int:
        v = s.get("document_id")
        if v is None:
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    sources = [
        SourceReference(
            document_id=_doc_id(s),
            filename=s.get("filename", "Unknown"),
            page=s.get("page"),
            chunk_text=s.get("chunk_text", ""),
            score=float(s.get("score") or 0.0),
        )
        for s in rag_response["sources"]
    ]

    # Get title from metadata.topic (refresh to get latest after title generation)
    db.refresh(conversation)  # Refresh from database
    metadata = getattr(conversation, "conversation_metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    conversation_title = metadata.get("topic") if metadata else None
    # Default to "Untitled" if no topic is set
    if not conversation_title:
        conversation_title = "Untitled"

    logger.info(
        "Reading conversation title after refresh",
        conversation_id=conversation_id,
        conversation_title=conversation_title,
        metadata=metadata,
    )

    logger.info(
        "=== RETURNING CHAT RESPONSE ===",
        conversation_id=conversation_id,
        conversation_title=conversation_title,
        topic_metadata=metadata.get("topic") if metadata else None,
        full_metadata=metadata,
        response_length=len(rag_response.get("response", "")),
    )

    return ChatResponse(
        conversation_id=conversation_id,
        conversation_title=conversation_title,  # Include updated title from metadata.topic
        message=MessageResponse(
            id=str(assistant_message.id),  # Convert to string
            session_id=str(
                conversation_id
            ),  # Use session_id instead of conversation_id
            role="assistant",
            content=rag_response["response"],
            sources=sources,
            input_tokens=rag_response.get("input_tokens"),
            output_tokens=rag_response.get("output_tokens"),
            created_at=assistant_message.created_at,
            feedback=None,
            parent_message_id=None,
            version_label=None,
            is_active_path=True,
            citations=[],
            translation_metadata=None,
            child_messages=[],
        ),
        sources=sources,
    )


# Quick chat endpoint (creates conversation automatically)
@router.post("/chat", response_model=ChatResponse)
async def quick_chat(
    chat_request: ChatRequest,
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Quick chat - creates a new conversation if needed."""
    # Get or create conversation
    if chat_request.conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == chat_request.conversation_id,
                Conversation.user_id == current_user.id,
            )
            .first()
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )
    else:
        # Verify agent exists
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

        # Create new conversation with metadata.topic
        default_title = f"Chat with {agent.name}"
        conversation = Conversation(
            conversation_metadata={
                "topic": default_title
            },  # Use conversation_metadata attribute name
            user_id=current_user.id,
            agent_id=agent_id,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Forward to the chat endpoint
    return await chat(conversation.id, chat_request, db, current_user)


# Feedback endpoints
@router.post("/messages/{message_id}/feedback", response_model=FeedbackResponse)
async def create_feedback(
    message_id: int,
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Add feedback to a message."""
    # Verify message belongs to user's conversation
    message = (
        db.query(Message)
        .join(Conversation)
        .filter(Message.id == message_id, Conversation.user_id == current_user.id)
        .first()
    )

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        )

    # Check if feedback already exists
    existing = db.query(Feedback).filter(Feedback.message_id == message_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback already exists for this message",
        )

    feedback = Feedback(
        message_id=message_id,
        thumbs_up=feedback_data.thumbs_up,
        comment=feedback_data.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    logger.info(
        "Created feedback", message_id=message_id, thumbs_up=feedback_data.thumbs_up
    )
    return feedback


@router.put("/messages/{message_id}/feedback", response_model=FeedbackResponse)
async def update_feedback(
    message_id: int,
    feedback_data: FeedbackUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update feedback on a message."""
    # Verify message belongs to user's conversation
    message = (
        db.query(Message)
        .join(Conversation)
        .filter(Message.id == message_id, Conversation.user_id == current_user.id)
        .first()
    )

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        )

    feedback = db.query(Feedback).filter(Feedback.message_id == message_id).first()
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
        )

    if feedback_data.thumbs_up is not None:
        feedback.thumbs_up = feedback_data.thumbs_up
    if feedback_data.comment is not None:
        feedback.comment = feedback_data.comment

    db.commit()
    db.refresh(feedback)

    return feedback

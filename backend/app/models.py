"""Database models for RAG Agent Management Platform."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Boolean,
    Float,
    JSON,
    Enum as SQLEnum,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy import Table
from .database import Base

# Association table for Agent <-> DataDomain many-to-many
agent_data_domains = Table(
    "agent_data_domains",
    Base.metadata,
    Column("agent_id", Integer, ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True),
    Column("data_domain_id", Integer, ForeignKey("data_domains.id", ondelete="CASCADE"), primary_key=True),
)


def generate_uuid():
    """Generate a UUID string."""
    return str(uuid.uuid4())


class AgentType(str, Enum):
    """Types of agents available."""

    RAG = "rag"
    TRANSLATION = "translation"
    VIDEO_TRANSCRIPTION = "video_transcription"


class AccessPolicy(str, Enum):
    """Access policy for chat sessions."""

    PRIVATE = "private"
    SHARED_LINK = "shared_link"
    ORGANIZATION_WIDE = "organization_wide"


class LifecycleState(str, Enum):
    """Lifecycle state for chat sessions."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    SOFT_DELETED = "soft_deleted"
    PURGED = "purged"


class RatingType(str, Enum):
    """Type of rating for feedback."""

    BINARY = "binary"  # thumbs up/down
    SCALE = "scale"  # 1-5 scale


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    data_domains = relationship("DataDomain", back_populates="owner")
    agents = relationship("Agent", back_populates="owner")
    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )


class DataDomain(Base):
    """Data Domain - a collection of documents for RAG."""

    __tablename__ = "data_domains"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    # Model name fetched dynamically from EMBEDDING_API_BASE/v1/models
    embedding_model = Column(String(255), nullable=False)
    qdrant_collection = Column(String(255), unique=True, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # VLM configuration for video processing (optional, falls back to global settings)
    vlm_api_base = Column(String(512), nullable=True)
    vlm_api_key = Column(String(512), nullable=True)
    vlm_model_id = Column(String(255), nullable=True)
    video_mode = Column(String(50), nullable=True)  # "procedure" or "race"
    vlm_prompt = Column(Text, nullable=True)  # Custom prompt; if empty, use default from prompts.py
    object_tracker = Column(String(50), nullable=True, default="yolo")  # "none", "simple_blob", or "yolo"
    enable_ocr = Column(Boolean, nullable=True, default=False)  # OCR on video frames

    # Relationships
    owner = relationship("User", back_populates="data_domains")
    documents = relationship(
        "Document", back_populates="data_domain", cascade="all, delete-orphan"
    )
    agents = relationship(
        "Agent", secondary=agent_data_domains, back_populates="data_domains"
    )


class Document(Base):
    """Document within a Data Domain."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50))  # pdf, txt, docx, etc.
    file_size = Column(Integer)  # in bytes
    chunk_count = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, processing, ready, failed
    error_message = Column(Text)
    processing_type = Column(String(50), default="document")  # "document" or "video"
    data_domain_id = Column(Integer, ForeignKey("data_domains.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    data_domain = relationship("DataDomain", back_populates="documents")


class Agent(Base):
    """Agent configuration - supports RAG and Translation agents."""

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    agent_type = Column(
        String(50),
        nullable=False,
        default=AgentType.RAG.value,
        index=True,
    )

    # Model name fetched dynamically from LLM_API_BASE/v1/models
    # Nullable for video transcription agents that don't use LLM
    llm_model = Column(String(255), nullable=True)
    system_prompt = Column(Text)  # Optional for some agent types
    temperature = Column(Float, default=0.7)
    top_p = Column(Float, default=1.0)
    top_k = Column(Integer, default=50)
    max_tokens = Column(Integer, default=2048)

    # RAG-specific settings (optional for translation agents)
    retrieval_k = Column(Integer, default=5)  # Number of chunks to retrieve

    # Translation-specific settings
    source_language = Column(String(50), default="auto")  # Auto-detect or specific
    target_language = Column(String(50), default="en")  # Target language code
    supported_languages = Column(JSON)  # List of supported language codes

    # Test data (optional)
    test_questions = Column(JSON)  # List of {question, reference_answer}

    # Video transcription-specific settings
    whisper_model_size = Column(
        String(50), default="base"
    )  # tiny, base, small, medium, large

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Unique constraint: each owner can only have one agent with a given name
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="agents_owner_id_name_unique"),
    )

    # Relationships
    owner = relationship("User", back_populates="agents")
    data_domains = relationship(
        "DataDomain", secondary=agent_data_domains, back_populates="agents"
    )
    # Use Conversation for now (ChatSession will replace it after migration)
    conversations = relationship(
        "Conversation", back_populates="agent", cascade="all, delete-orphan"
    )


class JobType(str, Enum):
    """Unified job type for queue + DB."""

    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    INGESTION_WEB = "ingestion_web"
    INGESTION_FILE = "ingestion_file"
    INGESTION_VIDEO = "ingestion_video"


class JobStatus(str, Enum):
    """Unified job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """
    Unified job record: Redis for queueing only, DB for all state.

    payload: type-specific input (see payload shapes below).
    result: type-specific output after completion (optional).
    """

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String(50), nullable=False, index=True)  # JobType value
    status = Column(String(50), nullable=False, default="pending", index=True)  # JobStatus value

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True, index=True)

    payload = Column(JSON, nullable=False)  # job parameters (type-specific)
    result = Column(JSON, nullable=True)  # job output after completion (type-specific)

    progress = Column(Integer, default=0)  # 0-100
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", backref="unified_jobs")
    agent = relationship("Agent", backref="unified_jobs")
    document = relationship("Document", backref="unified_jobs")


# Payload/result shapes per job_type (for reference; enforce in services):
#
# transcription:
#   payload: { "sub_type": "transcribe"|"add_subtitles", "source_path", "job_directory", "source_filename",
#              "language?", "subtitle_language?", "translation_agent_id?" }
#   result:  { "transcript"?, "segments"?, "language"?, "result_video_path"? }
#
# translation:
#   payload: { "source_file_path", "job_directory", "source_filename", "source_file_type",
#              "target_language", "simplified_mode" }
#   result:  { "output_filename", "output_file_path", "input_tokens", "output_tokens", "processing_time_seconds" }
#
# ingestion_web:
#   payload: { "url", "collection_name", "source_config", "processing_config", "metadata" }
#   result:  { "pages_processed", "chunks_created", "markdown"? }
#
# ingestion_file:
#   payload: { "file_path", "original_filename", "collection_name", "processing_config",
#              "document_id?" }
#   result:  { "chunks_created" }
#
# ingestion_video:
#   payload: { "file_path", "original_filename", "collection_name", "document_id?",
#              "embedding_model", "vlm_api_base?", "vlm_api_key?", "vlm_model_id?", "video_mode?" }
#   result:  { "chunks_created" }


class TranslationJob(Base):
    """Translation job for file translations."""

    __tablename__ = "translation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Job directory - all files for this job are stored here
    job_directory = Column(String(512))

    # Source file info
    source_filename = Column(String(255), nullable=False)
    source_file_path = Column(String(512), nullable=False)
    source_file_type = Column(String(50))

    # Translation settings
    target_language = Column(String(50), nullable=False)
    simplified_mode = Column(Boolean, default=False)

    # Output file info
    output_filename = Column(String(255))
    output_file_path = Column(String(512))

    # Job status
    status = Column(
        String(50), default="pending"
    )  # pending, extracting, translating, generating, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    error_message = Column(Text)

    # Extracted markdown (intermediate step)
    extracted_markdown = Column(Text)
    translated_markdown = Column(Text)

    # Metrics
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    processing_time_seconds = Column(Float)  # Total processing time in seconds

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)


class TranscriptionJob(Base):
    """Video transcription job (transcribe or add_subtitles)."""

    __tablename__ = "transcription_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)

    job_type = Column(String(50), nullable=False)  # transcribe | add_subtitles
    status = Column(
        String(50), default="pending"
    )  # pending, processing, completed, failed

    source_filename = Column(String(255), nullable=False)
    source_path = Column(String(512), nullable=False)
    job_directory = Column(String(512), nullable=False)  # relative to UPLOAD_DIR

    language = Column(String(50), nullable=True)
    subtitle_language = Column(String(50), nullable=True)
    translation_agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)

    # Result: transcript stored as JSON; video path for add_subtitles
    result_transcript = Column(
        JSON, nullable=True
    )  # {transcript, segments, language, ...}
    result_video_path = Column(String(512), nullable=True)  # path to final_video.mp4

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", backref="transcription_jobs")
    agent = relationship("Agent", foreign_keys=[agent_id])
    translation_agent = relationship("Agent", foreign_keys=[translation_agent_id])


# Conversation model
class Conversation(Base):
    """User conversation with an agent."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    conversation_metadata = Column(
        "metadata", JSON
    )  # OpenAI-compatible metadata: {"topic": "conversation title", ...}
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="conversations")
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """Message in a chat session - supports tree structure for versioning."""

    __tablename__ = "messages"

    # Primary key - using Integer to match existing schema
    # Will be changed to String(36) UUID after migration
    id = Column(Integer, primary_key=True, index=True)
    # id = Column(String(36), primary_key=True, default=generate_uuid, index=True)  # After migration

    # Session reference - using Integer to match existing conversations table
    conversation_id = Column(
        Integer, ForeignKey("conversations.id"), nullable=False, index=True
    )

    # Tree structure for versioning
    # NOTE: Using Integer for now to match existing messages.id
    # Will be changed to String(36) after migration
    parent_message_id = Column(
        Integer,
        ForeignKey("messages.id"),
        nullable=True,
        index=True,
        # String(36), ForeignKey("messages.id") - After migration
    )

    # Message content
    role = Column(String(20), nullable=False, index=True)  # user, assistant
    content = Column(Text, nullable=False)
    sources = Column(JSON)  # Legacy sources format for backward compatibility

    # Versioning
    version_label = Column(String(100))  # E.g., "Draft 1", "Draft 2" (for UI display)
    is_active_path = Column(
        Boolean, default=True, index=True
    )  # For exports - only export true path

    # Translation-specific metadata
    translation_metadata = Column(JSON)  # {source_lang, target_lang, simplified, etc.}

    # Token usage
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    parent_message = relationship("Message", remote_side=[id], backref="child_messages")
    # Temporarily disabled until migration
    # citations = relationship(
    #     "RAGCitation", back_populates="message", cascade="all, delete-orphan"
    # )
    feedback = relationship(
        "Feedback",
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
    )


class Feedback(Base):
    """User feedback on agent responses - Enhanced with rating types and tags."""

    __tablename__ = "feedback"

    # UUID primary key
    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)

    # Message reference
    # NOTE: Using Integer for now to match existing messages table
    # Will be changed to String(36) after migration
    message_id = Column(
        Integer,
        ForeignKey("messages.id"),
        nullable=False,
        unique=True,
        index=True,
        # String(36), ForeignKey("messages.id") - After migration
    )

    # Rating
    rating_type = Column(
        SQLEnum(RatingType), nullable=False, default=RatingType.BINARY, index=True
    )
    score = Column(Integer, nullable=True)  # 1-5 for scale, None for binary
    thumbs_up = Column(
        Boolean, nullable=True
    )  # For backward compatibility and binary ratings

    # Quality control
    tags = Column(JSON)  # List of strings: ["Hallucination", "Incomplete", "Harmful"]
    comment = Column(Text)  # Free text input from user

    # User who provided feedback
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    message = relationship("Message", back_populates="feedback")
    user = relationship("User", backref="feedback_given")

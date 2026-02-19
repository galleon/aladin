"""Pydantic schemas for RAG Agent Management Platform."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ============== Enums ==============


class AgentType(str, Enum):
    """Types of agents available."""

    RAG = "rag"
    TRANSLATION = "translation"
    VIDEO_TRANSCRIPTION = "video_transcription"
    AVATAR = "avatar"


# ============== Auth Schemas ==============


class Token(BaseModel):
    """JWT Token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload data."""

    user_id: int | None = None
    email: str | None = None


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
    full_name: str | None = None


class UserCreate(UserBase):
    """User creation schema."""

    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """User update schema."""

    email: EmailStr | None = None
    full_name: str | None = None
    password: str | None = Field(None, min_length=8)


class UserResponse(UserBase):
    """User response schema."""

    id: int
    is_active: bool
    is_superuser: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


# ============== Data Domain Schemas ==============


class DataDomainBase(BaseModel):
    """Base data domain schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    embedding_model: str = Field(
        ..., description="Embedding model ID from available models"
    )
    # VLM configuration for video processing (optional)
    vlm_api_base: str | None = Field(None, description="VLM API endpoint URL")
    vlm_api_key: str | None = Field(None, description="VLM API key")
    vlm_model_id: str | None = Field(None, description="VLM model ID")
    video_mode: str | None = Field(
        None, description="Video processing mode: 'procedure' or 'race'"
    )
    vlm_prompt: str | None = Field(None, description="Custom VLM prompt; empty uses default")
    object_tracker: str | None = Field(
        "yolo", description="Object tracker: 'yolo', 'yolo_api', 'simple_blob', or 'none'"
    )
    enable_ocr: bool | None = Field(False, description="Extract text from frames using OCR")


class DataDomainCreate(DataDomainBase):
    """Data domain creation schema."""

    pass


class DataDomainUpdate(BaseModel):
    """Data domain update schema."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    vlm_api_base: str | None = Field(None, description="VLM API endpoint URL")
    vlm_api_key: str | None = Field(None, description="VLM API key")
    vlm_model_id: str | None = Field(None, description="VLM model ID")
    video_mode: str | None = Field(
        None, description="Video processing mode: 'procedure' or 'race'"
    )
    vlm_prompt: str | None = Field(None, description="Custom VLM prompt; empty uses default")
    object_tracker: str | None = Field(None, description="Object tracker: 'yolo', 'yolo_api', 'simple_blob', or 'none'")
    enable_ocr: bool | None = Field(None, description="Extract text from frames using OCR")


class DocumentResponse(BaseModel):
    """Document response schema."""

    id: int
    filename: str
    original_filename: str
    file_type: str | None
    file_size: int | None
    chunk_count: int
    status: str
    error_message: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class DataDomainResponse(DataDomainBase):
    """Data domain response schema."""

    id: int
    qdrant_collection: str
    owner_id: int
    created_at: datetime
    updated_at: datetime
    documents: list[DocumentResponse] = []

    class Config:
        from_attributes = True


class VideoIngestionDefaults(BaseModel):
    """Default values for video ingestion (from env + prompt library); used by Data Domain create form."""

    vlm_api_base: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""
    default_prompt_procedure: str = ""
    default_prompt_race: str = ""
    prompt_library: dict[str, str] = {}  # keyword -> template (e.g. procedure, race)
    # True when VLM_API_BASE is set in env: backend has a default VLM for video segments (video mode "active")
    video_ingestion_available: bool = False


class DataDomainListResponse(BaseModel):
    """Data domain list response (without documents)."""

    id: int
    name: str
    description: str | None
    embedding_model: str
    qdrant_collection: str
    owner_id: int
    document_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class VectorChunkResponse(BaseModel):
    """Single vector store chunk (point) for inspect API."""

    id: str
    payload: dict


class ChunksListResponse(BaseModel):
    """Paginated list of vector store chunks (sorted by source_file, then chunk_index/t_start)."""

    items: list[VectorChunkResponse]
    total: int = 0
    has_more: bool = False
    next_offset: str | None = None


# ============== Agent Schemas ==============


class RAGAgentCreate(BaseModel):
    """RAG Agent creation schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    agent_type: AgentType = AgentType.RAG
    llm_model: str = Field(..., description="LLM model ID from available models")
    system_prompt: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    max_tokens: int = Field(2048, ge=1, le=16384)
    data_domain_ids: list[int] = Field(default_factory=list, description="Data domain IDs for RAG")
    retrieval_k: int = Field(5, ge=1, le=20)
    is_public: bool = False
    test_questions: list[dict] | None = None


class TranslationAgentCreate(BaseModel):
    """Translation Agent creation schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    agent_type: AgentType = AgentType.TRANSLATION
    llm_model: str = Field(..., description="LLM model ID from available models")
    system_prompt: str | None = Field(
        None,
        description="Optional custom system prompt. Use {target_language} and {simplified} as placeholders. If empty, built-in prompt is used.",
    )
    temperature: float = Field(0.3, ge=0.0, le=2.0)  # Lower temp for translation
    max_tokens: int = Field(4096, ge=1, le=16384)
    source_language: str = Field("auto", description="Source language code or 'auto'")
    target_language: str = Field("en", description="Target language code")
    supported_languages: list[str] | None = None
    is_public: bool = False


class VideoTranscriptionAgentCreate(BaseModel):
    """Video Transcription Agent creation schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    agent_type: AgentType | None = None  # Optional, will be set automatically
    whisper_model_size: str | None = Field(
        None,
        description="Whisper model size (optional, ignored for OpenAI API): tiny, base, small, medium, large",
    )
    is_public: bool = False


class AvatarAgentCreate(BaseModel):
    """Avatar Agent creation schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    llm_model: str = Field(..., description="LLM model ID from available models")
    system_prompt: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=16384)
    data_domain_ids: list[int] = Field(default_factory=list, description="Optional RAG data domain IDs")
    retrieval_k: int = Field(5, ge=1, le=20)
    avatar_config: dict | None = Field(
        None,
        description='Avatar config, e.g. {"video_source_url": "...", "image_url": "..."}',
    )
    is_public: bool = False


class AvatarSessionResponse(BaseModel):
    """Response containing a LiveKit room token for an avatar session."""

    room_name: str
    token: str
    livekit_url: str


class AgentUpdate(BaseModel):
    """Agent update schema (works for both types)."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    llm_model: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1, le=100)
    max_tokens: int | None = Field(None, ge=1, le=16384)
    data_domain_ids: list[int] | None = None
    retrieval_k: int | None = Field(None, ge=1, le=20)
    source_language: str | None = None
    target_language: str | None = None
    supported_languages: list[str] | None = None
    is_public: bool | None = None
    test_questions: list[dict] | None = None
    avatar_config: dict | None = None


class AgentResponse(BaseModel):
    """Agent response schema (unified for both types)."""

    id: int
    name: str
    description: str | None
    agent_type: AgentType
    llm_model: str | None  # Optional for video transcription agents
    system_prompt: str | None
    temperature: float | None  # Optional for video transcription agents
    top_p: float | None
    top_k: int | None
    max_tokens: int | None  # Optional for video transcription agents
    # RAG-specific
    data_domain_ids: list[int] = Field(default_factory=list)
    retrieval_k: int | None
    # Translation-specific
    source_language: str | None
    target_language: str | None
    supported_languages: list[str] | None
    # Avatar-specific
    avatar_config: dict | None = None
    # Common
    owner_id: int
    is_public: bool
    test_questions: list[dict] | None
    model_available: bool = True  # Whether the LLM model is currently available
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """Agent list response."""

    id: int
    name: str
    description: Optional[str] = None
    agent_type: AgentType
    llm_model: Optional[str] = None  # Optional for video transcription agents
    data_domain_ids: list[int] = Field(default_factory=list)
    target_language: Optional[str] = None
    owner_id: int
    is_public: bool
    model_available: bool = True  # Whether the LLM model is currently available
    created_at: datetime

    class Config:
        from_attributes = True


# ============== Translation Schemas ==============


class TranslationRequest(BaseModel):
    """Request to translate text."""

    text: str = Field(..., min_length=1)
    target_language: str = Field(..., description="Target language code")
    simplified: bool = Field(False, description="Use simplified language mode")
    source_language: str = Field("auto", description="Source language code or 'auto'")


class TranslationResponse(BaseModel):
    """Response containing translated text."""

    translated_text: str
    source_language: str
    target_language: str
    simplified: bool
    input_tokens: int
    output_tokens: int


class TranslationJobCreate(BaseModel):
    """Create a file translation job."""

    target_language: str = Field(..., description="Target language code")
    simplified: bool = Field(False, description="Use simplified language mode")


class TranslationJobResponse(BaseModel):
    """Translation job response."""

    id: int
    agent_id: int
    source_filename: str
    target_language: str
    simplified_mode: bool
    status: str
    progress: int
    error_message: str | None
    output_filename: str | None
    job_directory: str | None
    input_tokens: int | None
    output_tokens: int | None
    processing_time_seconds: float | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class SupportedLanguagesResponse(BaseModel):
    """List of supported languages."""

    languages: dict[str, str]  # {code: name}


# ============== Video Transcription Job Schemas ==============


class TranscriptionJobListResponse(BaseModel):
    """Transcription job list item."""

    id: int
    agent_id: int
    job_type: str  # transcribe | add_subtitles
    status: str
    source_filename: str
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class TranscriptionJobDetailResponse(BaseModel):
    """Transcription job detail with optional result."""

    id: int
    agent_id: int
    job_type: str
    status: str
    source_filename: str
    language: str | None
    subtitle_language: str | None
    error_message: str | None
    result_transcript: dict | None  # {transcript, segments, language, ...}
    result_video_path: str | None
    created_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class TranscriptionJobQueuedResponse(BaseModel):
    """Response when a transcription job is queued."""

    job_id: int
    message: str = "Job queued"


# ============== Unified Job Schemas ==============


class JobType(str, Enum):
    """Unified job type."""

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


class JobResponse(BaseModel):
    """Unified job response: job_id, job_type, job_status, payload, result."""

    id: int
    job_type: str
    status: str
    user_id: int
    agent_id: int | None = None
    document_id: int | None = None
    payload: dict  # type-specific parameters
    result: dict | None = None  # type-specific output after completion
    progress: int = 0
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


class JobCreate(BaseModel):
    """Create a unified job (payload is type-specific)."""

    job_type: JobType
    payload: dict = Field(..., description="Type-specific job parameters")
    agent_id: int | None = None
    document_id: int | None = None


# ============== Chat Session & Message Schemas ==============


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


class SourceReference(BaseModel):
    """Source reference in a RAG response (legacy - use RAGCitation)."""

    document_id: int
    filename: str
    page: int | None = None
    chunk_text: str
    score: float


class TranslationMetadata(BaseModel):
    """Metadata for translation messages."""

    source_language: str
    target_language: str
    simplified: bool


class RAGCitationResponse(BaseModel):
    """RAG Citation response schema."""

    id: str
    message_id: str
    document_id: int
    chunk_text: str
    vector_id: str | None = None
    similarity_score: float | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class MessageBase(BaseModel):
    """Base message schema."""

    role: str
    content: str


class MessageCreate(BaseModel):
    """Message creation schema (user query)."""

    content: str = Field(..., min_length=1)
    parent_message_id: str | None = (
        None  # For versioning - edit a question to create a new branch
    )


class MessageResponse(MessageBase):
    """Message response schema."""

    id: str
    session_id: str
    parent_message_id: str | None = None
    version_label: str | None = None
    is_active_path: bool = True
    citations: list[RAGCitationResponse] = []  # RAG citations
    sources: list[SourceReference] | None = (
        None  # Legacy format for backward compatibility
    )
    translation_metadata: TranslationMetadata | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    created_at: datetime
    feedback: Optional["FeedbackResponse"] = None
    child_messages: list["MessageResponse"] = []  # For versioning tree

    class Config:
        from_attributes = True


class ChatSessionBase(BaseModel):
    """Base chat session schema."""

    title: str | None = (
        None  # Deprecated: use metadata.topic instead. Kept for backward compatibility in API requests
    )
    metadata: dict[str, str] | None = (
        None  # OpenAI-compatible metadata (e.g., {"topic": "conversation title"})
    )
    agent_id: int
    tenant_id: int | None = None
    access_policy: AccessPolicy = AccessPolicy.PRIVATE
    allowed_users: list[int] = Field(default_factory=list)


class ChatSessionCreate(ChatSessionBase):
    """Chat session creation schema."""

    pass


class ChatSessionUpdate(BaseModel):
    """Chat session update schema."""

    title: str | None = (
        None  # Deprecated: use metadata.topic instead. Kept for backward compatibility
    )
    metadata: dict[str, str] | None = None  # OpenAI-compatible metadata
    access_policy: AccessPolicy | None = None
    allowed_users: list[int] | None = None
    lifecycle_state: LifecycleState | None = None


class ChatSessionResponse(ChatSessionBase):
    """Chat session response schema."""

    id: str
    owner_id: int
    lifecycle_state: LifecycleState
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []
    metadata: dict[str, str] | None = None  # Explicitly include metadata in response

    class Config:
        from_attributes = True


class ChatSessionListResponse(BaseModel):
    """Chat session list response."""

    id: (
        str | int
    )  # Support both UUID (new) and Integer (old) for backward compatibility
    title: str | None
    agent_id: int
    agent_name: str
    agent_type: AgentType | None = None  # Optional for backward compatibility
    owner_id: int | None = None  # Optional for backward compatibility
    access_policy: AccessPolicy | None = None  # Optional for backward compatibility
    lifecycle_state: LifecycleState | None = None  # Optional for backward compatibility
    message_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Backward compatibility aliases
ConversationBase = ChatSessionBase
ConversationCreate = ChatSessionCreate
ConversationUpdate = ChatSessionUpdate
ConversationResponse = ChatSessionResponse
ConversationListResponse = ChatSessionListResponse


# ============== Feedback Schemas ==============


class RatingType(str, Enum):
    """Type of rating for feedback."""

    BINARY = "binary"  # thumbs up/down
    SCALE = "scale"  # 1-5 scale


class FeedbackCreate(BaseModel):
    """Feedback creation schema."""

    rating_type: RatingType = RatingType.BINARY
    score: int | None = Field(None, ge=1, le=5, description="1-5 scale rating")
    thumbs_up: bool | None = Field(None, description="For binary ratings")
    tags: list[str] = Field(
        default_factory=list,
        description="Quality tags: Hallucination, Incomplete, Harmful, etc.",
    )
    comment: str | None = None


class FeedbackUpdate(FeedbackCreate):
    """Feedback update schema."""

    pass


class FeedbackResponse(BaseModel):
    """Feedback response schema."""

    id: str
    message_id: str
    rating_type: RatingType
    score: int | None = None
    thumbs_up: bool | None = None
    tags: list[str] = []
    comment: str | None = None
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============== Chat Schemas ==============


class ChatRequest(BaseModel):
    """Chat request schema for RAG agents."""

    message: str = Field(..., min_length=1)
    conversation_id: int | None = None


class TranslationChatRequest(BaseModel):
    """Chat request schema for translation agents."""

    message: str = Field(..., min_length=1)
    target_language: str = Field(..., description="Target language code")
    simplified: bool = Field(False, description="Use simplified language mode")
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    """Chat response schema."""

    conversation_id: int
    conversation_title: str | None = None
    message: MessageResponse
    sources: list[SourceReference] = []


# ============== Voice Schemas ==============


class VoiceTranscribeResponse(BaseModel):
    """Voice transcription response schema."""

    text: str = Field(..., description="Transcribed text from audio")
    language: str | None = Field(None, description="Detected language")


class VoiceTextToSpeechRequest(BaseModel):
    """Text-to-speech request schema."""

    text: str = Field(..., min_length=1, max_length=4096, description="Text to convert to speech")
    voice: str = Field("alloy", description="Voice to use (e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speed of speech (0.25 to 4.0)")


# Update forward references
MessageResponse.model_rebuild()

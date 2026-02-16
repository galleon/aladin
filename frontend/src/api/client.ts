/**
 * API Client for ALADIN
 */
import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

/** Default timeout so the UI never hangs waiting for a request (e.g. backend down). */
const API_TIMEOUT_MS = 20000;

// Create axios instance
const apiClient: AxiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: API_TIMEOUT_MS,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor to handle auth errors
apiClient.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('access_token');
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// ============== Types ==============

export type AgentType = 'rag' | 'translation' | 'video_transcription';

export interface User {
    id: number;
    email: string;
    full_name: string | null;
    is_active: boolean;
    is_superuser?: boolean;
    created_at: string;
}

export interface DataDomain {
    id: number;
    name: string;
    description: string | null;
    embedding_model: string;
    qdrant_collection: string;
    owner_id: number;
    document_count?: number;
    documents?: Document[];
    // VLM configuration for video processing (optional)
    vlm_api_base?: string | null;
    vlm_api_key?: string | null;
    vlm_model_id?: string | null;
    video_mode?: string | null; // "procedure" or "race"
    vlm_prompt?: string | null;
    object_tracker?: string | null; // "yolo", "simple_blob", or "none"
    enable_ocr?: boolean | null;
    created_at: string;
    updated_at: string;
}

export interface Document {
    id: number;
    filename: string;
    original_filename: string;
    file_type: string | null;
    file_size: number | null;
    chunk_count: number;
    status: string;
    error_message: string | null;
    processing_type?: string; // "document" or "video"
    created_at: string;
}

export interface Agent {
    id: number;
    name: string;
    description: string | null;
    agent_type: AgentType;
    llm_model: string | null; // Optional for video transcription agents
    system_prompt: string | null;
    temperature: number | null; // Optional for video transcription agents
    top_p: number | null;
    top_k: number | null;
    max_tokens: number | null; // Optional for video transcription agents
    // RAG-specific
    data_domain_ids: number[];
    retrieval_k: number | null;
    // Translation-specific
    source_language: string | null;
    target_language: string | null;
    supported_languages: string[] | null;
    // Common
    test_questions: Array<{ question: string; reference_answer: string }> | null;
    owner_id: number;
    is_public: boolean;
    model_available?: boolean; // Whether the LLM model is currently available
    created_at: string;
    updated_at: string;
}

export interface SourceReference {
    document_id: number;
    filename: string;
    page: number | null;
    chunk_text: string;
    score: number;
}

export interface TranslationMetadata {
    source_language: string;
    target_language: string;
    simplified: boolean;
}

export interface Message {
    id: number;
    conversation_id: number;
    role: 'user' | 'assistant';
    content: string;
    sources: SourceReference[] | null;
    translation_metadata: TranslationMetadata | null;
    input_tokens: number | null;
    output_tokens: number | null;
    created_at: string;
    feedback: Feedback | null;
}

export interface Conversation {
    id: number;
    title: string | null;
    user_id: number;
    agent_id: number;
    agent_name?: string;
    agent_type?: AgentType;
    message_count?: number;
    messages?: Message[];
    created_at: string;
    updated_at: string;
}

export interface Feedback {
    id: number;
    message_id: number;
    thumbs_up: boolean | null;
    comment: string | null;
    created_at: string;
}

export interface ChatResponse {
    conversation_id: number;
    message: Message;
    sources: SourceReference[];
}

export interface ModelInfo {
    id: string;
    name: string;
    owned_by: string | null;
    type: 'llm' | 'embedding';
}

export interface ModelsResponse {
    models: ModelInfo[];
    endpoint: string;
    error?: string | null;
}

export interface EndpointConfig {
    llm_base: string;
    embedding_base: string;
}

export interface TranslationResponse {
    translated_text: string;
    source_language: string;
    target_language: string;
    simplified: boolean;
    input_tokens: number;
    output_tokens: number;
}

export interface TranslationJob {
    id: number;
    agent_id: number;
    source_filename: string;
    target_language: string;
    simplified_mode: boolean;
    status: string;
    progress: number;
    error_message: string | null;
    output_filename: string | null;
    job_directory: string | null;
    input_tokens: number | null;
    output_tokens: number | null;
    processing_time_seconds: number | null;
    created_at: string;
    completed_at: string | null;
}

export interface SupportedLanguages {
    languages: Record<string, string>;
}

// ============== Auth API ==============

export const authApi = {
    register: (email: string, password: string, fullName?: string) =>
        apiClient.post<User>('/auth/register', { email, password, full_name: fullName }),

    login: (email: string, password: string) => {
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);
        return apiClient.post<{ access_token: string; token_type: string }>(
            '/auth/login',
            formData,
            { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
        );
    },

    getCurrentUser: () => apiClient.get<User>('/auth/me'),
};

// ============== Data Domains API ==============

export type VideoIngestionDefaults = {
    vlm_api_base: string;
    vlm_api_key: string;
    vlm_model: string;
    default_prompt_procedure: string;
    default_prompt_race: string;
    prompt_library: Record<string, string>;
    /** True when VLM_API_BASE is set in .env (video mode / VLM is active at backend level). */
    video_ingestion_available: boolean;
};

export const dataDomainsApi = {
    list: () => apiClient.get<DataDomain[]>('/data-domains/'),

    get: (id: number) => apiClient.get<DataDomain>(`/data-domains/${id}`),

    getVideoDefaults: () =>
        apiClient.get<VideoIngestionDefaults>('/data-domains/video-defaults').then((res) => res.data),

    create: (data: {
        name: string;
        description?: string;
        embedding_model: string;
        vlm_api_base?: string | null;
        vlm_api_key?: string | null;
        vlm_model_id?: string | null;
        video_mode?: string | null;
        vlm_prompt?: string | null;
        object_tracker?: string | null;
        enable_ocr?: boolean | null;
    }) => apiClient.post<DataDomain>('/data-domains/', data),

    update: (id: number, data: {
        name?: string;
        description?: string;
        vlm_api_base?: string | null;
        vlm_api_key?: string | null;
        vlm_model_id?: string | null;
        video_mode?: string | null;
        vlm_prompt?: string | null;
        object_tracker?: string | null;
        enable_ocr?: boolean | null;
    }) => apiClient.put<DataDomain>(`/data-domains/${id}`, data),

    delete: (id: number) => apiClient.delete(`/data-domains/${id}`),

    uploadDocument: (domainId: number, file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        return apiClient.post<Document>(`/data-domains/${domainId}/documents`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
    },

    deleteDocument: (domainId: number, documentId: number) =>
        apiClient.delete(`/data-domains/${domainId}/documents/${documentId}`),

    getChunks: (domainId: number, params?: { limit?: number; offset?: number }) =>
        apiClient.get<{
            items: { id: string; payload: Record<string, unknown> }[];
            total: number;
            has_more: boolean;
            next_offset: string | null;
        }>(`/data-domains/${domainId}/chunks`, { params: params ?? {} }),

    reindexDomain: (domainId: number) =>
        apiClient.post<{ message: string; documents_queued: number }>(`/data-domains/${domainId}/reindex`),

    reindexDocument: (domainId: number, documentId: number) =>
        apiClient.post<{ message: string; document_id: number }>(
            `/data-domains/${domainId}/documents/${documentId}/reindex`
        ),
};

// ============== Agents API ==============

export const agentsApi = {
    list: (agentType?: AgentType) =>
        apiClient.get<Agent[]>('/agents/', { params: agentType ? { agent_type: agentType } : {} }),

    get: (id: number) => apiClient.get<Agent>(`/agents/${id}`),

    createRAG: (data: {
        name: string;
        description?: string;
        llm_model: string;
        system_prompt: string;
        temperature?: number;
        top_p?: number;
        top_k?: number;
        max_tokens?: number;
        data_domain_ids: number[];
        retrieval_k?: number;
        is_public?: boolean;
    }) => apiClient.post<Agent>('/agents/rag', { ...data, agent_type: 'rag' }),

    createTranslation: (data: {
        name: string;
        description?: string;
        llm_model: string;
        system_prompt?: string | null;
        temperature?: number;
        max_tokens?: number;
        source_language?: string;
        target_language?: string;
        supported_languages?: string[];
        is_public?: boolean;
    }) => apiClient.post<Agent>('/agents/translation', { ...data, agent_type: 'translation' }),

    createVideoTranscription: (data: {
        name: string;
        description?: string;
        is_public?: boolean;
    }) => apiClient.post<Agent>('/agents/video-transcription', data),
    getVideoTranscriptionConfig: () => apiClient.get<{ available: boolean; whisper_api_base: string | null }>('/agents/video-transcription/config'),

    // Legacy - defaults to RAG
    create: (data: Partial<Agent>) => apiClient.post<Agent>('/agents/', data),

    update: (id: number, data: Partial<Agent>) => apiClient.put<Agent>(`/agents/${id}`, data),

    delete: (id: number) => apiClient.delete(`/agents/${id}`),

    test: (id: number) => apiClient.post<{ results: any[] }>(`/agents/${id}/test`),
};

// ============== Conversations API ==============

export const conversationsApi = {
    list: (agentId?: number) => {
        const params = agentId ? { agent_id: agentId } : {};
        return apiClient.get<Conversation[]>('/conversations/', { params });
    },

    get: (id: number) => apiClient.get<Conversation>(`/conversations/${id}`),

    create: (agentId: number, title?: string) =>
        apiClient.post<Conversation>('/conversations/', { agent_id: agentId, title }),

    update: (id: number, title: string) =>
        apiClient.put<Conversation>(`/conversations/${id}`, { title }),

    delete: (id: number) => apiClient.delete(`/conversations/${id}`),

    chat: (conversationId: number, message: string) =>
        apiClient.post<ChatResponse>(`/conversations/${conversationId}/chat`, { message }),

    quickChat: (agentId: number, message: string, conversationId?: number) =>
        apiClient.post<ChatResponse>(`/conversations/chat?agent_id=${agentId}`, {
            message,
            conversation_id: conversationId,
        }),

    addFeedback: (messageId: number, thumbsUp: boolean | null, comment?: string) =>
        apiClient.post<Feedback>(`/conversations/messages/${messageId}/feedback`, {
            thumbs_up: thumbsUp,
            comment,
        }),

    updateFeedback: (messageId: number, thumbsUp: boolean | null, comment?: string) =>
        apiClient.put<Feedback>(`/conversations/messages/${messageId}/feedback`, {
            thumbs_up: thumbsUp,
            comment,
        }),
};

// ============== Translation API ==============

export const translationApi = {
    getSupportedLanguages: () =>
        apiClient.get<SupportedLanguages>('/translation/languages'),

    translateText: (agentId: number, data: {
        text: string;
        target_language: string;
        simplified?: boolean;
        source_language?: string;
    }) => apiClient.post<TranslationResponse>(`/translation/${agentId}/translate`, data),

    chat: (agentId: number, data: {
        message: string;
        target_language: string;
        simplified?: boolean;
        conversation_id?: number;
    }) => apiClient.post<ChatResponse>(`/translation/${agentId}/chat`, data),

    translateFile: (agentId: number, file: File, targetLanguage: string, simplified = false) => {
        const formData = new FormData();
        formData.append('file', file);
        return apiClient.post<TranslationJob>(
            `/translation/${agentId}/translate-file?target_language=${targetLanguage}&simplified=${simplified}`,
            formData,
            { headers: { 'Content-Type': 'multipart/form-data' } }
        );
    },

    getJob: (jobId: number) => apiClient.get<TranslationJob>(`/translation/jobs/${jobId}`),

    listJobs: (agentId?: number, status?: string) => {
        const params: Record<string, any> = {};
        if (agentId) params.agent_id = agentId;
        if (status) params.status = status;
        return apiClient.get<TranslationJob[]>('/translation/jobs', { params });
    },

    downloadTranslation: (jobId: number) =>
        apiClient.get(`/translation/jobs/${jobId}/download`, { responseType: 'blob' }),
};

// ============== Video Transcription API ==============

export interface TranscriptionJobListItem {
    id: number;
    agent_id: number;
    job_type: string;
    status: string;
    source_filename: string;
    created_at: string;
    completed_at: string | null;
}

export interface TranscriptionJobDetail {
    id: number;
    agent_id: number;
    job_type: string;
    status: string;
    source_filename: string;
    language: string | null;
    subtitle_language: string | null;
    error_message: string | null;
    result_transcript: {
        transcript: string;
        segments: Array<{ start: number; end: number; text: string }>;
        language: string;
        language_probability?: number;
    } | null;
    result_video_path: string | null;
    created_at: string;
    completed_at: string | null;
}

export interface TranscriptionJobQueued {
    job_id: number;
    message: string;
}

export const videoTranscriptionApi = {
    transcribe: (agentId: number, video: File, language?: string) => {
        const formData = new FormData();
        formData.append('video', video);
        return apiClient.post<TranscriptionJobQueued>(
            `/video-transcription/transcribe?agent_id=${agentId}${language ? `&language=${language}` : ''}`,
            formData,
            { headers: { 'Content-Type': 'multipart/form-data' } }
        );
    },

    addSubtitles: (agentId: number, video: File, language?: string, subtitleLanguage?: string, translationAgentId?: number) => {
        const formData = new FormData();
        formData.append('video', video);
        const params = new URLSearchParams();
        params.append('agent_id', agentId.toString());
        if (language) params.append('language', language);
        if (subtitleLanguage) params.append('subtitle_language', subtitleLanguage);
        if (translationAgentId) params.append('translation_agent_id', translationAgentId.toString());
        return apiClient.post<TranscriptionJobQueued>(
            `/video-transcription/add-subtitles?${params.toString()}`,
            formData,
            { headers: { 'Content-Type': 'multipart/form-data' } }
        );
    },

    listJobs: (agentId?: number) =>
        apiClient.get<TranscriptionJobListItem[]>(
            '/video-transcription/jobs',
            { params: agentId != null ? { agent_id: agentId } : undefined }
        ),

    getQueueStatus: () =>
        apiClient.get<{ queue_name: string; queued_count: number | null; error?: string }>(
            '/video-transcription/queue-status',
            { timeout: 5000 }
        ),

    getQueueDebug: () =>
        apiClient.get<{
            queue_name: string;
            queued_count: number;
            job_ids_sample: string[];
            error?: string;
        }>('/video-transcription/queue-debug', { timeout: 5000 }),

    listQueues: () =>
        apiClient.get<{ queues: Array<{ name: string; label: string; count: number | null }>; error?: string }>(
            '/video-transcription/queues'
        ),

    purgeQueue: (queueName: string) =>
        apiClient.post<{ queue_name: string; removed: number }>(
            `/video-transcription/queues/purge?queue_name=${encodeURIComponent(queueName)}`
        ),

    deleteJob: (jobId: number) =>
        apiClient.delete(`/video-transcription/jobs/${jobId}`),

    requeueJob: (jobId: number) =>
        apiClient.post<{ job_id: number; message: string }>(`/video-transcription/jobs/${jobId}/requeue`),

    purgeUserJobs: () =>
        apiClient.delete<{ deleted: number }>('/video-transcription/jobs'),

    getJob: (jobId: number) =>
        apiClient.get<TranscriptionJobDetail>(`/video-transcription/jobs/${jobId}`),

    getResult: (jobId: number) =>
        apiClient.get<{ transcript: string; segments: Array<{ start: number; end: number; text: string }>; language: string; language_probability?: number }>(
            `/video-transcription/jobs/${jobId}/result`
        ),

    downloadVideo: (jobId: number) =>
        apiClient.get(`/video-transcription/jobs/${jobId}/download`, { responseType: 'blob' }),
};

// ============== Unified Jobs API (admin: queue status + jobs table) ==============

export interface UnifiedJobListItem {
    id: number;
    job_type: string;
    status: string;
    created_at: string;
    completed_at: string | null;
    duration_seconds: number | null;
}

export interface QueueStatusResponse {
    queue_name: string;
    queued_count: number | null;
    error?: string;
}

export interface QueueInfo {
    name: string;
    label: string;
    count: number | null;
}

export const jobsApi = {
    listJobs: (params?: { job_type?: string; status?: string; limit?: number }) =>
        apiClient.get<UnifiedJobListItem[]>('/jobs', { params: params ?? {}, timeout: 10000 }),

    getQueueStatus: () =>
        apiClient.get<QueueStatusResponse>('/jobs/queue-status', { timeout: 5000 }),

    listQueues: () =>
        apiClient.get<{ queues: QueueInfo[]; error?: string }>('/jobs/queues', { timeout: 5000 }),

    purgeQueue: (queueName: string) =>
        apiClient.post<{ queue_name: string; removed: number }>(
            `/jobs/queues/purge?queue_name=${encodeURIComponent(queueName)}`
        ),

    requeue: (jobId: number) =>
        apiClient.post<{ job_id: number; message: string }>(`/jobs/${jobId}/requeue`),

    cancel: (jobId: number) =>
        apiClient.patch<{ job_id: number; message: string }>(`/jobs/${jobId}/cancel`),

    delete: (jobId: number) =>
        apiClient.delete(`/jobs/${jobId}`),
};

// ============== Ingestion API (data domain uploads, web/file jobs) ==============

export interface IngestionJob {
    job_id: string;
    status: string;
    progress: number;
    message?: string;
    created_at: string;
    started_at?: string;
    completed_at?: string;
    pages_processed: number;
    pages_total: number;
    chunks_created: number;
    error_message?: string;
    type?: string;
    filename?: string;
}

export const ingestionApi = {
    listJobs: (status?: string, limit = 50) =>
        apiClient.get<IngestionJob[]>('/ingestion/jobs', {
            params: { limit, ...(status ? { status } : {}) },
            timeout: 10000,
        }),
    cancelJob: (jobId: string) =>
        apiClient.delete<{ message: string }>(`/ingestion/jobs/${encodeURIComponent(jobId)}`),
};

// ============== Stats API ==============

export interface UsageStats {
    prompts_count: number;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
}

export const statsApi = {
    getUsage: () => apiClient.get<UsageStats>('/stats/usage'),
};

// ============== Models API ==============

export const modelsApi = {
    listLLM: (refresh = false) =>
        apiClient.get<ModelsResponse>('/models/llm', { params: { refresh } }),

    listEmbedding: (refresh = false) =>
        apiClient.get<ModelsResponse>('/models/embedding', { params: { refresh } }),

    getConfig: () => apiClient.get<EndpointConfig>('/models/config'),

    refresh: () => apiClient.post<{ message: string; llm_count: number; embedding_count: number }>('/models/refresh'),
};

// ============== Voice API ==============

export interface VoiceTranscribeResponse {
    text: string;
    language: string | null;
}

export const voiceApi = {
    transcribe: (audioFile: Blob, language?: string) => {
        const formData = new FormData();
        formData.append('file', audioFile, 'recording.webm');
        if (language) {
            formData.append('language', language);
        }
        return apiClient.post<VoiceTranscribeResponse>('/voice/transcribe', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 60000, // 60 seconds for transcription
        });
    },

    synthesize: (text: string, voice = 'alloy', speed = 1.0) => {
        return apiClient.post<Blob>('/voice/synthesize', 
            { text, voice, speed },
            { 
                responseType: 'blob',
                timeout: 60000, // 60 seconds for synthesis
            }
        );
    },
};

export default apiClient;

import axios from 'axios'

const api = axios.create({
    baseURL: '/api',
    headers: { 'Content-Type': 'application/json' },
})

// Response interceptor to handle errors
api.interceptors.response.use(
    (response) => response,
    (error) => {
        // Log network errors for debugging
        if (error.message === 'Network Error' || error.code === 'ERR_NETWORK') {
            console.error('Network Error:', error)
            // Check if it's a CORS issue or connection issue
            if (!error.response) {
                error.message = 'Unable to connect to the server. Please check your connection.'
            }
        }
        return Promise.reject(error)
    }
)

// Auth
export const login = (email: string, password: string) => {
    const formData = new URLSearchParams()
    formData.append('username', email)
    formData.append('password', password)
    return api.post<{ access_token: string }>('/auth/login', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
}

export const register = (email: string, password: string, fullName?: string) =>
    api.post('/auth/register', { email, password, full_name: fullName })

export const getMe = () => api.get('/auth/me')

// Agents
export const getAgents = () => api.get('/agents/')
export const getAgent = (id: number) => api.get(`/agents/${id}/`)

// Conversations
export const getConversations = (agentId?: number) =>
    api.get('/conversations/', { params: agentId ? { agent_id: agentId } : {} })

export const getConversation = (id: number) => api.get(`/conversations/${id}`)

export const createConversation = (agentId: number) =>
    api.post('/conversations/', { agent_id: agentId })

export const chat = (conversationId: number, message: string) =>
    api.post(`/conversations/${conversationId}/chat`, { message })

export const submitFeedback = (messageId: number, thumbsUp: boolean, comment?: string) =>
    api.post(`/conversations/messages/${messageId}/feedback`, {
        thumbs_up: thumbsUp,
        comment
    })

// Data Domains
export const getDataDomains = () => api.get('/data-domains/')

export const uploadDocument = (domainId: number, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/data-domains/${domainId}/documents`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })
}

// Translation
export const translateText = (agentId: number, data: {
    text: string
    target_language: string
    simplified?: boolean
    source_language?: string
}) => api.post(`/translation/${agentId}/translate`, data)

export const translateFile = (agentId: number, file: File, targetLanguage: string, simplified: boolean) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('target_language', targetLanguage)
    formData.append('simplified', String(simplified))
    return api.post(`/translation/${agentId}/translate-file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })
}

export const getTranslationJobs = (agentId: number) =>
    api.get(`/translation/jobs?agent_id=${agentId}`)

export const downloadTranslation = (jobId: number) =>
    api.get(`/translation/jobs/${jobId}/download`, { responseType: 'blob' })

// Video Transcription
export const transcribeVideo = (agentId: number, video: File, language?: string) => {
    const formData = new FormData()
    formData.append('video', video)
    return api.post(`/video-transcription/transcribe?agent_id=${agentId}${language ? `&language=${language}` : ''}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })
}

export const addSubtitlesToVideo = (agentId: number, video: File, language?: string, subtitleLanguage?: string, translationAgentId?: number) => {
    const formData = new FormData()
    formData.append('video', video)
    const params = new URLSearchParams()
    params.append('agent_id', agentId.toString())
    if (language) params.append('language', language)
    if (subtitleLanguage) params.append('subtitle_language', subtitleLanguage)
    if (translationAgentId) params.append('translation_agent_id', translationAgentId.toString())
    return api.post(`/video-transcription/add-subtitles?${params.toString()}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
    })
}

export default api



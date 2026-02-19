import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import api from './api'

interface User {
    id: number
    email: string
    full_name?: string
}

interface Agent {
    id: number
    name: string
    description?: string
    llm_model: string | null
    temperature: number | null
    agent_type?: 'rag' | 'translation' | 'video_transcription' | 'avatar'
    target_language?: string
    model_available?: boolean // Whether the LLM model is currently available
    avatar_config?: {
        video_source_url?: string
        image_url?: string
    } | null
}

interface Conversation {
    id: number
    title: string
    agent_id: number
    message_count?: number
}

export interface Message {
    id?: number
    role: 'user' | 'assistant'
    content: string
    sources?: Array<{
        filename: string
        page?: number
        score?: number
    }>
}

interface DataDomain {
    id: number
    name: string
    document_count?: number
}

interface AppState {
    // Auth
    token: string | null
    user: User | null
    setAuth: (token: string, user: User) => void
    logout: () => void

    // Agents
    agents: Agent[]
    selectedAgent: Agent | null
    setAgents: (agents: Agent[]) => void
    selectAgent: (agent: Agent | null) => void

    // Conversations
    conversations: Conversation[]
    selectedConversation: Conversation | null
    setConversations: (convs: Conversation[] | ((prev: Conversation[]) => Conversation[])) => void
    selectConversation: (conv: Conversation | null) => void

    // Messages
    messages: Message[]
    setMessages: (msgs: Message[] | ((prev: Message[]) => Message[])) => void
    addMessage: (msg: Message) => void
    updateLastMessage: (content: string) => void

    // Data Domains
    domains: DataDomain[]
    setDomains: (domains: DataDomain[]) => void

    // Loading
    isLoading: boolean
    setLoading: (loading: boolean) => void
}

export const useStore = create<AppState>()(
    persist(
        (set) => ({
            // Auth
            token: null,
            user: null,
            setAuth: (token, user) => {
                api.defaults.headers.common['Authorization'] = `Bearer ${token}`
                set({ token, user })
            },
            logout: () => {
                delete api.defaults.headers.common['Authorization']
                set({
                    token: null,
                    user: null,
                    agents: [],
                    selectedAgent: null,
                    conversations: [],
                    selectedConversation: null,
                    messages: [],
                    domains: [],
                })
            },

            // Agents
            agents: [],
            selectedAgent: null,
            setAgents: (agents) => set({ agents }),
            selectAgent: (agent) => set({
                selectedAgent: agent,
                selectedConversation: null,
                messages: [],
            }),

            // Conversations
            conversations: [],
            selectedConversation: null,
            setConversations: (conversationsOrUpdater) => {
                if (typeof conversationsOrUpdater === 'function') {
                    set((state) => ({
                        conversations: conversationsOrUpdater(state.conversations),
                    }))
                } else {
                    set({ conversations: conversationsOrUpdater })
                }
            },
            selectConversation: (conv) => set({ selectedConversation: conv }),

            // Messages
            messages: [],
            setMessages: (messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])) => {
                if (typeof messagesOrUpdater === 'function') {
                    set((state) => ({
                        messages: messagesOrUpdater(state.messages),
                    }))
                } else {
                    set({ messages: messagesOrUpdater })
                }
            },
            addMessage: (msg) => {
                set((state) => ({
                    messages: [...state.messages, msg],
                }))
            },
            updateLastMessage: (content) => {
                set((state) => ({
                    messages: state.messages.map((m, i) =>
                        i === state.messages.length - 1 ? { ...m, content } : m
                    ),
                }))
            },

            // Data Domains
            domains: [],
            setDomains: (domains) => set({ domains }),

            // Loading
            isLoading: false,
            setLoading: (isLoading) => set({ isLoading }),
        }),
        {
            name: 'chat-storage',
            partialize: (state) => ({ token: state.token, user: state.user }),
        }
    )
)

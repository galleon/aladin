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
    agent_type?: 'rag' | 'translation' | 'video_transcription'
    target_language?: string
    model_available?: boolean // Whether the LLM model is currently available
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
                    // Functional update
                    set((state) => {
                        const newConversations = conversationsOrUpdater(state.conversations)
                        console.log('Store: setConversations (functional) called, updating from', state.conversations.length, 'to', newConversations.length, 'conversations')
                        return { conversations: newConversations }
                    })
                } else {
                    // Direct update
                    console.log('Store: setConversations called with', conversationsOrUpdater.length, 'conversations')
                    set({ conversations: conversationsOrUpdater })
                }
            },
            selectConversation: (conv) => set({ selectedConversation: conv }),

            // Messages
            messages: [],
            setMessages: (messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])) => {
                if (typeof messagesOrUpdater === 'function') {
                    // Functional update
                    set((state) => {
                        const newMessages = messagesOrUpdater(state.messages)
                        console.log('Store: setMessages (functional) called, updating from', state.messages.length, 'to', newMessages.length, 'messages')
                        return { messages: newMessages }
                    })
                } else {
                    // Direct update
                    console.log('Store: setMessages called with', messagesOrUpdater.length, 'messages')
                    set({ messages: messagesOrUpdater })
                }
            },
            addMessage: (msg) => {
                console.log('Store: addMessage called with', msg.role, 'message:', msg.content?.substring(0, 50))
                set((state) => {
                    const newMessages = [...state.messages, msg]
                    console.log('Store: messages after addMessage:', newMessages.length, 'messages')
                    return { messages: newMessages }
                })
            },
            updateLastMessage: (content) => {
                console.log('Store: updateLastMessage called with content:', content?.substring(0, 100))
                set((state) => {
                    const updated = state.messages.map((m, i) =>
                        i === state.messages.length - 1 ? { ...m, content } : m
                    )
                    console.log('Store: messages after updateLastMessage:', updated.length, 'messages')
                    return { messages: updated }
                })
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


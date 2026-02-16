import { useState, useRef, useEffect } from 'react'
import {
    Bot, Send, LogOut, MessageSquare, Plus, ChevronDown,
    ThumbsUp, ThumbsDown, FileText, Upload, X, Loader2,
    Sparkles, AlertCircle, ChevronRight
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { useStore } from '../store'
import {
    getConversations, getConversation, createConversation,
    chat, submitFeedback, uploadDocument
} from '../api'
import TranslationView from './TranslationView'
import VideoTranscriptionView from './VideoTranscriptionView'

export default function ChatView() {
    const {
        user, agents, selectedAgent, selectAgent,
        conversations, selectedConversation, selectConversation, setConversations,
        messages, setMessages, addMessage, updateLastMessage,
        domains, logout, isLoading, setLoading
    } = useStore()

    const [input, setInput] = useState('')
    const [showAgents, setShowAgents] = useState(false)
    const [showUpload, setShowUpload] = useState(false)
    const [uploadDomain, setUploadDomain] = useState<number | null>(null)
    const [uploadStatus, setUploadStatus] = useState('')
    const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const fileInputRef = useRef<HTMLInputElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Load conversations when an agent is selected (including via URL parameter)
    useEffect(() => {
        if (selectedAgent) {
            // Clear list immediately so we don't show previous agent's conversations while loading
            setConversations([])
            const loadConversations = async () => {
                try {
                    const res = await getConversations(selectedAgent.id)
                    setConversations(res.data || [])
                } catch {
                    // Failed to load conversations
                }
            }
            loadConversations()
        } else {
            setConversations([])
        }
    }, [selectedAgent?.id]) // Only re-run when agent ID changes

    const handleSelectAgent = async (agent: typeof agents[0]) => {
        selectAgent(agent)
        setShowAgents(false)
        try {
            const res = await getConversations(agent.id)
            setConversations(res.data || [])
        } catch {
            // Failed to load conversations
        }
    }

    const handleSelectConversation = async (conv: typeof conversations[0] | null) => {
        if (!conv) {
            selectConversation(null)
            setMessages([])
            return
        }
        // Only load messages if switching to a different conversation
        if (selectedConversation?.id !== conv.id) {
            selectConversation(conv)
            try {
                const res = await getConversation(conv.id)
                setMessages(res.data.messages || [])
            } catch {
                // Failed to load conversation messages
            }
        } else {
            // Same conversation, just update the reference
            selectConversation(conv)
        }
    }

    const handleSend = async () => {
        if (!input.trim() || !selectedAgent || isLoading) return

        const userMessage = input.trim()
        setInput('')

        addMessage({ role: 'user', content: userMessage })
        addMessage({ role: 'assistant', content: '...' })
        setLoading(true)

        try {
            let convId: number = selectedConversation?.id ?? 0
            if (!convId) {
                const res = await createConversation(selectedAgent.id)
                convId = res.data.id
                selectConversation(res.data)
                setConversations([res.data, ...conversations])
            }

            const res = await chat(convId, userMessage)
            const { message, sources, conversation_title } = res.data

            // Get messages from store to ensure we have the latest state
            // We need to update the last message with the response
            // Since we can't use functional updates, we'll use updateLastMessage first, then setMessages
            updateLastMessage(message.content)

            // Now update with ID and sources - we need to get the current messages from the store
            // Use a small delay to ensure updateLastMessage has processed
            setTimeout(() => {
                // Get the latest messages from the store by reading it
                // Since we can't directly read from store in component, we'll use the messages from the hook
                // But messages might be stale, so we'll construct the updated array manually
                // The messages should have been updated by updateLastMessage above
                setMessages((prev: typeof messages) => {
                    const updatedMessages = prev.map((m: typeof messages[0], i: number) => {
                        if (i === prev.length - 1) {
                            return { ...m, id: message.id, sources }
                        }
                        return m
                    })
                    return updatedMessages
                })
            }, 0)

            // Update conversation title if provided in response (generated during chat call)
            // Always update the conversations list, and selectedConversation if it exists
            let updatedConvs: typeof conversations = conversations

            if (conversation_title !== undefined && conversation_title !== null) {
                // Update in conversations list (always do this)
                // Check if conversation exists in list, if not add it
                const convExists = conversations.some((c: typeof conversations[0]) => {
                    const cId = typeof c.id === 'string' ? parseInt(c.id) : c.id
                    return cId === convId
                })

                if (convExists) {
                    // Update existing conversation
                    updatedConvs = conversations.map((c: typeof conversations[0]) => {
                        const cId = typeof c.id === 'string' ? parseInt(c.id) : c.id
                        return cId === convId ? { ...c, title: conversation_title } : c
                    })
                } else {
                    // Add new conversation to list (in case it wasn't there)
                    updatedConvs = [{ id: convId, title: conversation_title, agent_id: selectedAgent.id }, ...conversations]
                }
                setConversations(updatedConvs)

                // Always ensure the current conversation is selected (don't create new one)
                const updatedConv = updatedConvs.find((c: typeof conversations[0]) => {
                    const cId = typeof c.id === 'string' ? parseInt(c.id) : c.id
                    return cId === convId
                })
                if (updatedConv) {
                    // Update selectedConversation with the updated title
                    selectConversation(updatedConv)
                }
            }

            // Final safety check: ensure we're still on the same conversation after all updates
            // This prevents the UI from switching to a new conversation or losing selection
            if (updatedConvs) {
                const currentConv = updatedConvs.find((c: typeof conversations[0]) => {
                    const cId = typeof c.id === 'string' ? parseInt(c.id) : c.id
                    return cId === convId
                })
                if (currentConv) {
                    // Use setTimeout to ensure state updates are processed
                    setTimeout(() => {
                        selectConversation(currentConv)
                    }, 0)
                }
            }
        } catch (e: any) {
            updateLastMessage(`Error: ${e.response?.data?.detail || e.message}`)
        } finally {
            setLoading(false)
        }
    }

    const handleFeedback = async (messageId: number, isPositive: boolean) => {
        try {
            await submitFeedback(messageId, isPositive)
        } catch {
            // Failed to submit feedback
        }
    }

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file || !uploadDomain) return

        setUploadStatus('Uploading...')
        try {
            await uploadDocument(uploadDomain, file)
            setUploadStatus('✓ Uploaded successfully!')
            setTimeout(() => {
                setShowUpload(false)
                setUploadStatus('')
            }, 2000)
        } catch (err: any) {
            setUploadStatus(`Error: ${err.response?.data?.detail || err.message}`)
        }
    }

    // Render video transcription UI for video transcription agents
    if (selectedAgent?.agent_type === 'video_transcription') {
        return <VideoTranscriptionView />
    }

    // Render translation UI for translation agents
    if (selectedAgent?.agent_type === 'translation') {
        return (
            <div className="h-screen flex">
                {/* Sidebar */}
                <div className="w-72 bg-black/20 border-r border-white/10 flex flex-col">
                    {/* Header */}
                    <div className="p-4 border-b border-white/10">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                                <Bot className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <h1 className="font-semibold text-white">Translation</h1>
                                <p className="text-xs text-gray-500">{user?.full_name || user?.email}</p>
                            </div>
                        </div>
                    </div>

                    {/* Agent Selector */}
                    <div className="p-4 border-b border-white/10">
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">
                            Agent
                        </label>
                        <button
                            onClick={() => setShowAgents(!showAgents)}
                            className="w-full p-3 bg-white/5 rounded-xl border border-white/10 text-left flex items-center justify-between hover:bg-white/10 transition-colors"
                        >
                            <span className={selectedAgent ? 'text-white' : 'text-gray-500'}>
                                {selectedAgent?.name || 'Select an agent...'}
                            </span>
                            <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${showAgents ? 'rotate-180' : ''}`} />
                        </button>

                        {showAgents && (
                            <div className="mt-2 bg-gray-900 rounded-xl border border-white/10 overflow-hidden">
                                {agents.map((agent) => {
                                    const isUnavailable = agent.model_available === false
                                    return (
                                        <button
                                            key={agent.id}
                                            onClick={() => !isUnavailable && handleSelectAgent(agent)}
                                            disabled={isUnavailable}
                                            className={`w-full p-3 text-left transition-colors ${isUnavailable
                                                    ? 'opacity-50 cursor-not-allowed text-gray-500'
                                                    : selectedAgent?.id === agent.id
                                                        ? 'bg-emerald-500/20 text-emerald-300'
                                                        : 'text-gray-300 hover:bg-white/5'
                                                }`}
                                            title={isUnavailable ? `Model '${agent.llm_model}' is not available` : undefined}
                                        >
                                            <div className="flex items-center justify-between">
                                                <div className="font-medium">{agent.name}</div>
                                                {isUnavailable && (
                                                    <AlertCircle className="w-4 h-4 text-red-400" />
                                                )}
                                            </div>
                                            <div className="text-xs text-gray-500 mt-0.5">
                                                {agent.llm_model}
                                                {isUnavailable && ' (Unavailable)'}
                                            </div>
                                        </button>
                                    )
                                })}
                                {agents.length === 0 && (
                                    <div className="p-4 text-center text-gray-500 text-sm">
                                        No agents available
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Bottom Actions */}
                    <div className="mt-auto p-4 border-t border-white/10 space-y-2">
                        <button
                            onClick={() => setShowUpload(true)}
                            className="w-full p-3 rounded-xl bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10 transition-colors flex items-center gap-2"
                        >
                            <Upload className="w-4 h-4" />
                            Upload Documents
                        </button>
                        <button
                            onClick={logout}
                            className="w-full p-3 rounded-xl bg-white/5 border border-white/10 text-gray-300 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/30 transition-colors flex items-center gap-2"
                        >
                            <LogOut className="w-4 h-4" />
                            Logout
                        </button>
                    </div>
                </div>

                {/* Translation View */}
                <div className="flex-1">
                    <TranslationView />
                </div>

                {/* Upload Modal */}
                {showUpload && (
                    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
                        <div className="bg-gray-900 rounded-2xl p-6 w-full max-w-md border border-white/10">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-lg font-semibold text-white">Upload Document</h3>
                                <button onClick={() => setShowUpload(false)} className="text-gray-500 hover:text-white">
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="space-y-4">
                                <div>
                                    <label className="text-sm text-gray-400 mb-2 block">Data Domain</label>
                                    <select
                                        value={uploadDomain || ''}
                                        onChange={(e) => setUploadDomain(Number(e.target.value))}
                                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-emerald-500"
                                    >
                                        <option value="">Select a domain...</option>
                                        {domains.map((d) => (
                                            <option key={d.id} value={d.id}>{d.name}</option>
                                        ))}
                                    </select>
                                </div>

                                <div>
                                    <label className="text-sm text-gray-400 mb-2 block">File</label>
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept=".pdf,.txt,.md,.docx,.csv,.json"
                                        onChange={handleFileUpload}
                                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-emerald-500 file:text-white file:cursor-pointer"
                                    />
                                </div>

                                {uploadStatus && (
                                    <div className={`p-3 rounded-xl text-sm ${uploadStatus.includes('Error')
                                            ? 'bg-red-500/10 text-red-400'
                                            : 'bg-emerald-500/10 text-emerald-400'
                                        }`}>
                                        {uploadStatus}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        )
    }

    // Render chat UI for RAG agents
    return (
        <div className="h-screen flex">
            {/* Sidebar */}
            <div className="w-72 bg-black/20 border-r border-white/10 flex flex-col">
                {/* Header */}
                <div className="p-4 border-b border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                            <Bot className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h1 className="font-semibold text-white">ALADIN</h1>
                            <p className="text-xs text-gray-500">{user?.full_name || user?.email}</p>
                        </div>
                    </div>
                </div>

                {/* Agent Selector */}
                <div className="p-4 border-b border-white/10">
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2 block">
                        Agent
                    </label>
                    <button
                        onClick={() => setShowAgents(!showAgents)}
                        className="w-full p-3 bg-white/5 rounded-xl border border-white/10 text-left flex items-center justify-between hover:bg-white/10 transition-colors"
                    >
                        <span className={selectedAgent ? 'text-white' : 'text-gray-500'}>
                            {selectedAgent?.name || 'Select an agent...'}
                        </span>
                        <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${showAgents ? 'rotate-180' : ''}`} />
                    </button>

                    {showAgents && (
                        <div className="mt-2 bg-gray-900 rounded-xl border border-white/10 overflow-hidden">
                            {agents.map((agent) => (
                                <button
                                    key={agent.id}
                                    onClick={() => handleSelectAgent(agent)}
                                    className={`w-full p-3 text-left hover:bg-white/5 transition-colors ${selectedAgent?.id === agent.id ? 'bg-indigo-500/20 text-indigo-300' : 'text-gray-300'
                                        }`}
                                >
                                    <div className="font-medium">{agent.name}</div>
                                    <div className="text-xs text-gray-500 mt-0.5">{agent.llm_model}</div>
                                </button>
                            ))}
                            {agents.length === 0 && (
                                <div className="p-4 text-center text-gray-500 text-sm">
                                    No agents available
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Conversations */}
                <div className="flex-1 overflow-y-auto p-4">
                    <div className="flex items-center justify-between mb-3">
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Conversations
                        </label>
                        {selectedAgent && (
                            <button
                                onClick={() => handleSelectConversation(null)}
                                className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30 transition-colors"
                            >
                                <Plus className="w-4 h-4" />
                            </button>
                        )}
                    </div>

                    <div className="space-y-2">
                        {(() => {
                            const forCurrentAgent = selectedAgent
                                ? conversations.filter((conv) => Number(conv.agent_id) === selectedAgent.id)
                                : []
                            if (forCurrentAgent.length === 0) {
                                return (
                                    <div className="p-4 text-center text-gray-500 text-sm">
                                        {selectedAgent ? 'No conversations yet' : 'Select an agent to view conversations'}
                                    </div>
                                )
                            }
                            return forCurrentAgent.map((conv) => (
                                <button
                                    key={conv.id}
                                    onClick={() => handleSelectConversation(conv)}
                                    className={`w-full p-3 rounded-xl text-left transition-colors ${selectedConversation?.id === conv.id
                                            ? 'bg-indigo-500/20 border border-indigo-500/30'
                                            : 'bg-white/5 border border-transparent hover:bg-white/10'
                                        }`}
                                >
                                    <div className="flex items-center gap-2 min-w-0">
                                        <MessageSquare className="w-4 h-4 text-gray-500 flex-shrink-0" />
                                        <span className="text-sm text-white truncate flex-1" title={conv.title || 'Untitled'}>
                                            {conv.title || 'Untitled'}
                                        </span>
                                    </div>
                                </button>
                            ))
                        })()}
                    </div>
                </div>

                {/* Bottom Actions */}
                <div className="p-4 border-t border-white/10 space-y-2">
                    <button
                        onClick={() => setShowUpload(true)}
                        className="w-full p-3 rounded-xl bg-white/5 border border-white/10 text-gray-300 hover:bg-white/10 transition-colors flex items-center gap-2"
                    >
                        <Upload className="w-4 h-4" />
                        Upload Documents
                    </button>
                    <button
                        onClick={logout}
                        className="w-full p-3 rounded-xl bg-white/5 border border-white/10 text-gray-300 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/30 transition-colors flex items-center gap-2"
                    >
                        <LogOut className="w-4 h-4" />
                        Logout
                    </button>
                </div>
            </div>

            {/* Chat Area */}
            <div className="flex-1 flex flex-col">
                {/* Chat Header */}
                {selectedAgent && (
                    <div className="p-4 border-b border-white/10 bg-black/20">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                                <Sparkles className="w-5 h-5 text-white" />
                            </div>
                            <div className="min-w-0 flex-1">
                                <h2 className="font-semibold text-white truncate">{selectedAgent.name}</h2>
                                <p className="text-xs text-gray-500 truncate">
                                    {selectedAgent.llm_model}
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {!selectedAgent ? (
                        <div className="h-full flex flex-col items-center justify-center text-center">
                            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 flex items-center justify-center mb-4">
                                <Bot className="w-8 h-8 text-indigo-400" />
                            </div>
                            <h3 className="text-xl font-semibold text-white mb-2">Welcome to ALADIN</h3>
                            <p className="text-gray-500 max-w-sm">
                                Select an agent from the sidebar to start chatting with your documents.
                            </p>
                        </div>
                    ) : messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-center">
                            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 flex items-center justify-center mb-4">
                                <MessageSquare className="w-8 h-8 text-indigo-400" />
                            </div>
                            <h3 className="text-xl font-semibold text-white mb-2">Start a conversation</h3>
                            <p className="text-gray-500 max-w-sm">
                                Ask {selectedAgent.name} anything about your documents.
                            </p>
                        </div>
                    ) : (
                        (() => {
                            return messages.map((msg, idx) => (
                                    <div
                                        key={idx}
                                        className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}
                                    >
                                        {msg.role === 'assistant' && (
                                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                                                <Bot className="w-4 h-4 text-white" />
                                            </div>
                                        )}
                                        <div
                                            className={`max-w-[70%] rounded-2xl px-4 py-3 ${msg.role === 'user'
                                                    ? 'bg-indigo-500 text-white'
                                                    : 'bg-white/5 border border-white/10 text-gray-200'
                                                }`}
                                        >
                                            {msg.content === '...' ? (
                                                <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
                                            ) : (
                                                <>
                                                    <div className="chat-message prose prose-invert prose-sm max-w-none">
                                                        <ReactMarkdown
                                                            remarkPlugins={[remarkGfm]}
                                                            rehypePlugins={[rehypeHighlight]}
                                                        >
                                                            {msg.content}
                                                        </ReactMarkdown>
                                                    </div>
                                                    {msg.sources && msg.sources.length > 0 && (() => {
                                                        // Deduplicate sources by filename and page (frontend safety check)
                                                        const uniqueSources = msg.sources.filter((s, index, self) =>
                                                            index === self.findIndex((other) =>
                                                                other.filename === s.filename && other.page === s.page
                                                            )
                                                        )

                                                        return (
                                                            <div className="mt-3 pt-3 border-t border-white/10">
                                                                <button
                                                                    onClick={() => {
                                                                        const newExpanded = new Set(expandedSources)
                                                                        if (newExpanded.has(msg.id || 0)) {
                                                                            newExpanded.delete(msg.id || 0)
                                                                        } else {
                                                                            newExpanded.add(msg.id || 0)
                                                                        }
                                                                        setExpandedSources(newExpanded)
                                                                    }}
                                                                    className="flex items-center gap-2 text-xs text-gray-400 hover:text-gray-300 transition-colors mb-2"
                                                                >
                                                                    <ChevronRight
                                                                        className={`w-3 h-3 transition-transform ${expandedSources.has(msg.id || 0) ? 'rotate-90' : ''
                                                                            }`}
                                                                    />
                                                                    <span>Sources ({uniqueSources.length})</span>
                                                                </button>
                                                                {expandedSources.has(msg.id || 0) && (
                                                                    <div className="space-y-1.5">
                                                                        {uniqueSources.map((s, i) => (
                                                                            <div key={i} className="flex items-center gap-2 text-xs text-indigo-300 pl-5">
                                                                                <FileText className="w-3 h-3 flex-shrink-0" />
                                                                                <span className="truncate">{s.filename}</span>
                                                                                {s.page && <span className="text-gray-500 flex-shrink-0">p.{s.page}</span>}
                                                                                {s.score && (
                                                                                    <span className="text-gray-500 flex-shrink-0">
                                                                                        {(s.score * 100).toFixed(0)}%
                                                                                    </span>
                                                                                )}
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        )
                                                    })()}
                                                    {msg.role === 'assistant' && msg.id && (
                                                        <div className="mt-3 pt-3 border-t border-white/10 flex gap-2">
                                                            <button
                                                                onClick={() => handleFeedback(msg.id!, true)}
                                                                className="p-1.5 rounded-lg hover:bg-white/10 text-gray-500 hover:text-green-400 transition-colors"
                                                            >
                                                                <ThumbsUp className="w-4 h-4" />
                                                            </button>
                                                            <button
                                                                onClick={() => handleFeedback(msg.id!, false)}
                                                                className="p-1.5 rounded-lg hover:bg-white/10 text-gray-500 hover:text-red-400 transition-colors"
                                                            >
                                                                <ThumbsDown className="w-4 h-4" />
                                                            </button>
                                                        </div>
                                                    )}
                                                </>
                                            )}
                                        </div>
                                    </div>
                                ))
                        })()
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                {selectedAgent && (
                    <div className="p-4 border-t border-white/10 bg-black/20">
                        <div className="flex gap-3">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                                placeholder="Type your message..."
                                className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                            />
                            <button
                                onClick={handleSend}
                                disabled={!input.trim() || isLoading}
                                className="px-4 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                            >
                                <Send className="w-5 h-5" />
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Upload Modal */}
            {showUpload && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
                    <div className="bg-gray-900 rounded-2xl p-6 w-full max-w-md border border-white/10">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold text-white">Upload Document</h3>
                            <button onClick={() => setShowUpload(false)} className="text-gray-500 hover:text-white">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="text-sm text-gray-400 mb-2 block">Data Domain</label>
                                <select
                                    value={uploadDomain || ''}
                                    onChange={(e) => setUploadDomain(Number(e.target.value))}
                                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-indigo-500"
                                >
                                    <option value="">Select a domain...</option>
                                    {domains.map((d) => (
                                        <option key={d.id} value={d.id}>{d.name}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="text-sm text-gray-400 mb-2 block">File</label>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept=".pdf,.txt,.md,.docx,.csv,.json"
                                    onChange={handleFileUpload}
                                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-500 file:text-white file:cursor-pointer"
                                />
                            </div>

                            {uploadStatus && (
                                <div className={`p-3 rounded-xl text-sm ${uploadStatus.includes('Error')
                                        ? 'bg-red-500/10 text-red-400'
                                        : 'bg-indigo-500/10 text-indigo-400'
                                    }`}>
                                    {uploadStatus}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}


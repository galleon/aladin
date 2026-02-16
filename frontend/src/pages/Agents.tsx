import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { agentsApi, dataDomainsApi, modelsApi, AgentType } from '../api/client';
import { Bot, Plus, Trash2, MessageSquare, Loader2, X, Sparkles, AlertCircle, RefreshCw, Languages, Database, Video } from 'lucide-react';
import { useAuth } from '../App';

export default function Agents() {
    const { user } = useAuth();
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [agentTypeFilter, setAgentTypeFilter] = useState<AgentType | 'all'>('all');
    const [createAgentType, setCreateAgentType] = useState<AgentType>('rag');

    // RAG agent form state
    const [ragAgent, setRagAgent] = useState({
        name: '',
        description: '',
        llm_model: '',
        system_prompt: 'You are a helpful assistant that answers questions based on the provided context. Always cite your sources.',
        temperature: 0.7,
        top_p: 1.0,
        top_k: 50,
        data_domain_ids: [] as number[],
        retrieval_k: 5,
    });

    // Translation agent form state
    const [translationAgent, setTranslationAgent] = useState({
        name: '',
        description: '',
        llm_model: '',
        system_prompt: '' as string,
        temperature: 0.3,
        max_tokens: 4096,
    });

    // Video transcription agent form state
    const [videoTranscriptionAgent, setVideoTranscriptionAgent] = useState({
        name: '',
        description: '',
    });

    const queryClient = useQueryClient();

    // Fetch available LLM models (non-blocking, with retry and timeout)
    const { data: modelsData, isLoading: modelsLoading, error: modelsError, refetch: refetchModels } = useQuery({
        queryKey: ['llmModels'],
        queryFn: () => modelsApi.listLLM().then(res => res.data),
        retry: 1, // Only retry once
        retryDelay: 1000, // Wait 1 second before retry
        staleTime: 60000, // Cache for 60 seconds
        gcTime: 300000, // Keep in cache for 5 minutes
    });

    // Set default model when models are loaded
    useEffect(() => {
        if (modelsData?.models && modelsData.models.length > 0) {
            if (!ragAgent.llm_model) {
                setRagAgent(prev => ({ ...prev, llm_model: modelsData.models[0].id }));
            }
            if (!translationAgent.llm_model) {
                setTranslationAgent(prev => ({ ...prev, llm_model: modelsData.models[0].id }));
            }
        }
    }, [modelsData]);

    const { data: agents, isLoading } = useQuery({
        queryKey: ['agents', agentTypeFilter],
        queryFn: () => agentsApi.list(agentTypeFilter === 'all' ? undefined : agentTypeFilter).then(res => res.data),
    });

    const { data: dataDomains } = useQuery({
        queryKey: ['dataDomains'],
        queryFn: () => dataDomainsApi.list().then(res => res.data),
    });

    // Check if video transcription is available
    const { data: videoTranscriptionConfig } = useQuery({
        queryKey: ['videoTranscriptionConfig'],
        queryFn: () => agentsApi.getVideoTranscriptionConfig().then(res => res.data),
    });

    const createRAGMutation = useMutation({
        mutationFn: (data: typeof ragAgent) => agentsApi.createRAG(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['agents'] });
            setShowCreateModal(false);
            resetForms();
        },
    });

    const createTranslationMutation = useMutation({
        mutationFn: (data: typeof translationAgent) => agentsApi.createTranslation(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['agents'] });
            setShowCreateModal(false);
            resetForms();
        },
    });

    const createVideoTranscriptionMutation = useMutation({
        mutationFn: (data: typeof videoTranscriptionAgent) => agentsApi.createVideoTranscription(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['agents'] });
            setShowCreateModal(false);
            resetForms();
        },
        onError: (error: any) => {
            console.error('Failed to create video transcription agent:', error);
            const errorMessage = error?.response?.data?.detail || error?.message || 'Failed to create video transcription agent';
            alert(`Error: ${errorMessage}`);
        },
    });

    const deleteMutation = useMutation({
        mutationFn: (id: number) => agentsApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['agents'] });
        },
        onError: (error: any) => {
            const message = error?.response?.data?.detail || error?.message || 'Failed to delete agent';
            alert(`Error: ${message}`);
        },
    });

    const resetForms = () => {
        setRagAgent({
            name: '',
            description: '',
            llm_model: modelsData?.models[0]?.id || '',
            system_prompt: 'You are a helpful assistant that answers questions based on the provided context. Always cite your sources.',
            temperature: 0.7,
            top_p: 1.0,
            top_k: 50,
            data_domain_ids: [],
            retrieval_k: 5,
        });
        setTranslationAgent({
            name: '',
            description: '',
            llm_model: modelsData?.models[0]?.id || '',
            system_prompt: '',
            temperature: 0.3,
            max_tokens: 4096,
        });
        setVideoTranscriptionAgent({
            name: '',
            description: '',
        });
    };

    const handleCreate = (e: React.FormEvent) => {
        e.preventDefault();
        if (createAgentType === 'rag') {
            if (!ragAgent.data_domain_ids?.length) {
                alert('Please select at least one data domain');
                return;
            }
            if (!ragAgent.llm_model) {
                alert('Please select an LLM model');
                return;
            }
            createRAGMutation.mutate(ragAgent);
        } else if (createAgentType === 'translation') {
            if (!translationAgent.llm_model) {
                alert('Please select an LLM model');
                return;
            }
            createTranslationMutation.mutate(translationAgent);
        } else if (createAgentType === 'video_transcription') {
            createVideoTranscriptionMutation.mutate(videoTranscriptionAgent);
        }
    };

    const getAgentIcon = (type: AgentType) => {
        if (type === 'video_transcription') return Video;
        if (type === 'translation') return Languages;
        return Sparkles;
    };

    const getChatLink = (agent: { id: number; agent_type: AgentType }) => {
        // For both RAG and translation agents, open chat UI with agent selected
        const chatUiUrl = import.meta.env.VITE_CHAT_UI_URL || 'http://localhost:7860';
        return `${chatUiUrl}?agent_id=${agent.id}`;
    };

    return (
        <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Agents</h1>
                    <p className="text-slate-400">Configure and manage your AI agents</p>
                </div>
                <button
                    onClick={() => {
                        // Set agent type based on current filter, or default to 'rag'
                        if (agentTypeFilter && agentTypeFilter !== 'all') {
                            setCreateAgentType(agentTypeFilter);
                        } else {
                            setCreateAgentType('rag');
                        }
                        setShowCreateModal(true);
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all"
                >
                    <Plus className="w-5 h-5" />
                    Create Agent
                </button>
            </div>

            {/* Filter tabs */}
            <div className="flex gap-2">
                <button
                    onClick={() => setAgentTypeFilter('all')}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${agentTypeFilter === 'all'
                        ? 'bg-violet-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                        }`}
                >
                    All Agents
                </button>
                <button
                    onClick={() => setAgentTypeFilter('rag')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${agentTypeFilter === 'rag'
                        ? 'bg-violet-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                        }`}
                >
                    <Database className="w-4 h-4" />
                    RAG
                </button>
                <button
                    onClick={() => setAgentTypeFilter('translation')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${agentTypeFilter === 'translation'
                        ? 'bg-violet-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                        }`}
                >
                    <Languages className="w-4 h-4" />
                    Translation
                </button>
                <button
                    onClick={() => setAgentTypeFilter('video_transcription')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${agentTypeFilter === 'video_transcription'
                        ? 'bg-violet-600 text-white'
                        : 'bg-slate-800 text-slate-400 hover:text-white'
                        }`}
                >
                    <Bot className="w-4 h-4" />
                    Video
                </button>
            </div>

            {/* Loading */}
            {isLoading && (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
                </div>
            )}

            {/* Agents grid */}
            {agents && agents.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {agents.map((agent) => {
                        const Icon = getAgentIcon(agent.agent_type);
                        return (
                            <div
                                key={agent.id}
                                className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6 hover:border-violet-500/30 transition-all"
                            >
                                <div className="flex items-start justify-between mb-4">
                                    <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${agent.agent_type === 'video_transcription'
                                        ? 'bg-gradient-to-br from-blue-500/20 to-cyan-500/20'
                                        : agent.agent_type === 'translation'
                                            ? 'bg-gradient-to-br from-emerald-500/20 to-teal-500/20'
                                            : 'bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20'
                                        }`}>
                                        <Icon className={`w-6 h-6 ${agent.agent_type === 'video_transcription' ? 'text-blue-400' : agent.agent_type === 'translation' ? 'text-emerald-400' : 'text-violet-400'
                                            }`} />
                                    </div>
                                    <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <a
                                            href={getChatLink(agent)}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="p-2 rounded-lg text-slate-500 hover:bg-violet-500/10 hover:text-violet-400 transition-colors"
                                            title={agent.agent_type === 'translation' ? 'Translate' : 'Chat'}
                                        >
                                            <MessageSquare className="w-5 h-5" />
                                        </a>
                                        {user && agent.owner_id === user.id && (
                                            <button
                                                onClick={() => {
                                                    if (confirm('Delete this agent?')) {
                                                        deleteMutation.mutate(agent.id);
                                                    }
                                                }}
                                                className="p-2 rounded-lg text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-colors"
                                                title="Delete"
                                            >
                                                <Trash2 className="w-5 h-5" />
                                            </button>
                                        )}
                                    </div>
                                </div>
                                <Link to={`/agents/${agent.id}`}>
                                    <h3 className="text-lg font-semibold text-white mb-1 hover:text-violet-300 transition-colors">
                                        {agent.name}
                                    </h3>
                                </Link>
                                <p className="text-slate-400 text-sm mb-4 line-clamp-2">
                                    {agent.description || 'No description'}
                                </p>
                                <div className="flex items-center justify-between text-sm">
                                    <div className="flex items-center gap-2">
                                        <span className={`px-2 py-1 rounded-lg text-xs font-medium ${agent.agent_type === 'translation'
                                            ? 'bg-emerald-500/10 text-emerald-400'
                                            : agent.agent_type === 'video_transcription'
                                                ? 'bg-blue-500/10 text-blue-400'
                                                : 'bg-violet-500/10 text-violet-400'
                                            }`}>
                                            {agent.agent_type === 'translation' ? 'Translation' : agent.agent_type === 'video_transcription' ? 'Video' : 'RAG'}
                                        </span>
                                        {agent.target_language && (
                                            <span className="px-2 py-1 rounded-lg bg-slate-800/50 text-slate-400 text-xs">
                                                → {agent.target_language.toUpperCase()}
                                            </span>
                                        )}
                                    </div>
                                    <a
                                        href={getChatLink(agent)}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-violet-400 hover:text-violet-300 font-medium"
                                    >
                                        {agent.agent_type === 'translation' ? 'Translate →' : agent.agent_type === 'video_transcription' ? 'Transcribe →' : 'Chat →'}
                                    </a>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Empty state */}
            {agents && agents.length === 0 && (
                <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-12 text-center">
                    <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20">
                        <Bot className="w-8 h-8 text-violet-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">No agents yet</h3>
                    <p className="text-slate-400 mb-6">Create your first agent to get started.</p>
                    <button
                        onClick={() => {
                            // Set agent type based on current filter, or default to 'rag'
                            if (agentTypeFilter && agentTypeFilter !== 'all') {
                                setCreateAgentType(agentTypeFilter);
                            } else {
                                setCreateAgentType('rag');
                            }
                            setShowCreateModal(true);
                        }}
                        className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all"
                    >
                        <Plus className="w-5 h-5" />
                        Create Agent
                    </button>
                </div>
            )}

            {/* Create Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm overflow-y-auto">
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 w-full max-w-lg my-8 max-h-[90vh] overflow-y-auto">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-semibold text-white">Create Agent</h2>
                            <button
                                onClick={() => setShowCreateModal(false)}
                                className="p-2 rounded-lg text-slate-400 hover:bg-slate-800"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Agent Type Selector */}
                        <div className="flex gap-2 mb-6">
                            <button
                                type="button"
                                onClick={() => setCreateAgentType('rag')}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-colors ${createAgentType === 'rag'
                                    ? 'bg-violet-600 text-white'
                                    : 'bg-slate-800 text-slate-400 hover:text-white'
                                    }`}
                            >
                                <Database className="w-5 h-5" />
                                RAG Agent
                            </button>
                            <button
                                type="button"
                                onClick={() => setCreateAgentType('translation')}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-colors ${createAgentType === 'translation'
                                    ? 'bg-emerald-600 text-white'
                                    : 'bg-slate-800 text-slate-400 hover:text-white'
                                    }`}
                            >
                                <Languages className="w-5 h-5" />
                                Translation
                            </button>
                            <button
                                type="button"
                                onClick={() => setCreateAgentType('video_transcription')}
                                disabled={!videoTranscriptionConfig?.available}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium transition-colors ${!videoTranscriptionConfig?.available
                                    ? 'bg-slate-800/50 text-slate-600 cursor-not-allowed'
                                    : createAgentType === 'video_transcription'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-slate-800 text-slate-400 hover:text-white'
                                    }`}
                                title={!videoTranscriptionConfig?.available ? 'Video transcription requires WHISPER_API_BASE to be configured' : ''}
                            >
                                <Bot className="w-5 h-5" />
                                Video
                            </button>
                        </div>

                        <form onSubmit={handleCreate} className="space-y-4">
                            {/* Common fields */}
                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">Name</label>
                                <input
                                    type="text"
                                    value={createAgentType === 'rag'
                                        ? ragAgent.name
                                        : createAgentType === 'translation'
                                            ? translationAgent.name
                                            : videoTranscriptionAgent.name}
                                    onChange={(e) => {
                                        if (createAgentType === 'rag') {
                                            setRagAgent({ ...ragAgent, name: e.target.value });
                                        } else if (createAgentType === 'translation') {
                                            setTranslationAgent({ ...translationAgent, name: e.target.value });
                                        } else {
                                            setVideoTranscriptionAgent({ ...videoTranscriptionAgent, name: e.target.value });
                                        }
                                    }}
                                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                    placeholder={createAgentType === 'rag' ? 'My RAG Assistant' : createAgentType === 'translation' ? 'My Translator' : 'My Video Transcriber'}
                                    required
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">Description (optional)</label>
                                <textarea
                                    value={createAgentType === 'rag'
                                        ? ragAgent.description
                                        : createAgentType === 'translation'
                                            ? translationAgent.description
                                            : videoTranscriptionAgent.description}
                                    onChange={(e) => {
                                        if (createAgentType === 'rag') {
                                            setRagAgent({ ...ragAgent, description: e.target.value });
                                        } else if (createAgentType === 'translation') {
                                            setTranslationAgent({ ...translationAgent, description: e.target.value });
                                        } else {
                                            setVideoTranscriptionAgent({ ...videoTranscriptionAgent, description: e.target.value });
                                        }
                                    }}
                                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                    rows={2}
                                />
                            </div>

                            {/* Video Transcription specific fields */}
                            {createAgentType === 'video_transcription' && (
                                <div>
                                    {!videoTranscriptionConfig?.available && (
                                        <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                                            <div className="flex items-center gap-2 text-yellow-400">
                                                <AlertCircle className="w-5 h-5" />
                                                <span className="font-medium">Video Transcription Unavailable</span>
                                            </div>
                                            <p className="text-sm text-yellow-300/80 mt-2">
                                                WHISPER_API_BASE must be configured in the backend environment variables to create video transcription agents.
                                            </p>
                                        </div>
                                    )}
                                    {videoTranscriptionConfig?.available && (
                                        <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                                            <div className="flex items-center gap-2 text-blue-400">
                                                <Bot className="w-5 h-5" />
                                                <span className="font-medium">Whisper API Configured</span>
                                            </div>
                                            <p className="text-sm text-blue-300/80 mt-2">
                                                Using Whisper API at: {videoTranscriptionConfig.whisper_api_base || 'configured endpoint'}
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* LLM Model selector (not shown for video transcription) */}
                            {createAgentType !== 'video_transcription' && (
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <label className="block text-sm font-medium text-slate-300">LLM Model</label>
                                        <button
                                            type="button"
                                            onClick={() => refetchModels()}
                                            className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1"
                                        >
                                            <RefreshCw className="w-3 h-3" />
                                            Refresh
                                        </button>
                                    </div>

                                    {modelsLoading && (
                                        <div className="flex items-center gap-2 text-slate-400 text-sm py-3">
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Loading models...
                                        </div>
                                    )}

                                    {modelsError && (
                                        <div className="flex flex-col gap-2 text-red-400 py-4 bg-red-500/10 px-4 rounded-xl border border-red-500/20">
                                            <div className="flex items-center gap-2 font-medium">
                                                <AlertCircle className="w-5 h-5" />
                                                <span>Failed to connect to LLM endpoint</span>
                                            </div>
                                            <p className="text-sm text-red-300/80">
                                                Could not load models from the configured API endpoint. Please check that your LLM server is running.
                                            </p>
                                            <button
                                                type="button"
                                                onClick={() => refetchModels()}
                                                className="mt-2 text-sm text-red-300 hover:text-red-200 underline self-start"
                                            >
                                                Try again
                                            </button>
                                        </div>
                                    )}

                                    {modelsData && modelsData.models.length === 0 && (
                                        <div className="flex flex-col gap-2 text-amber-400 py-4 bg-amber-500/10 px-4 rounded-xl border border-amber-500/20">
                                            <div className="flex items-center gap-2 font-medium">
                                                <AlertCircle className="w-5 h-5" />
                                                <span>No LLM models available</span>
                                            </div>
                                            <p className="text-sm text-amber-300/80">
                                                {modelsData.error
                                                    ? <>Error: <code className="bg-amber-500/20 px-1 rounded">{modelsData.error}</code></>
                                                    : <>No models were found at <code className="bg-amber-500/20 px-1 rounded">{modelsData.endpoint}</code></>
                                                }
                                            </p>
                                            <button
                                                type="button"
                                                onClick={() => refetchModels()}
                                                className="mt-2 text-sm text-amber-300 hover:text-amber-200 underline self-start"
                                            >
                                                Refresh models
                                            </button>
                                        </div>
                                    )}

                                    {modelsData && modelsData.models.length > 0 && (
                                        <select
                                            value={createAgentType === 'rag' ? ragAgent.llm_model : translationAgent.llm_model}
                                            onChange={(e) => createAgentType === 'rag'
                                                ? setRagAgent({ ...ragAgent, llm_model: e.target.value })
                                                : setTranslationAgent({ ...translationAgent, llm_model: e.target.value })
                                            }
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                            required
                                        >
                                            {modelsData.models.map((model) => (
                                                <option key={model.id} value={model.id}>
                                                    {model.name}
                                                </option>
                                            ))}
                                        </select>
                                    )}
                                </div>
                            )}

                            {/* RAG-specific fields */}
                            {createAgentType === 'rag' && (
                                <>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-2">Data Domains</label>
                                        {dataDomains && dataDomains.length === 0 ? (
                                            <div className="text-amber-400 text-sm py-3 bg-amber-500/10 px-3 rounded-lg">
                                                You need to <Link to="/data-domains" className="underline">create a data domain</Link> first.
                                            </div>
                                        ) : (
                                            <div className="space-y-2 max-h-48 overflow-y-auto rounded-xl border border-slate-700/50 bg-slate-800/50 px-4 py-3">
                                                {dataDomains?.map((domain) => (
                                                    <label key={domain.id} className="flex items-center gap-2 cursor-pointer text-slate-200 hover:text-white">
                                                        <input
                                                            type="checkbox"
                                                            checked={ragAgent.data_domain_ids?.includes(domain.id) ?? false}
                                                            onChange={(e) => {
                                                                const ids = e.target.checked
                                                                    ? [...(ragAgent.data_domain_ids || []), domain.id]
                                                                    : (ragAgent.data_domain_ids || []).filter((id) => id !== domain.id);
                                                                setRagAgent({ ...ragAgent, data_domain_ids: ids });
                                                            }}
                                                            className="rounded border-slate-600 bg-slate-800 text-violet-500 focus:ring-violet-500"
                                                        />
                                                        <span>{domain.name}</span>
                                                    </label>
                                                ))}
                                            </div>
                                        )}
                                        {ragAgent.data_domain_ids?.length ? (
                                            <p className="text-xs text-slate-500 mt-1">Selected: {ragAgent.data_domain_ids.length} domain(s)</p>
                                        ) : null}
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-2">System Prompt</label>
                                        <textarea
                                            value={ragAgent.system_prompt}
                                            onChange={(e) => setRagAgent({ ...ragAgent, system_prompt: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                            rows={3}
                                            required
                                        />
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-2">Temperature</label>
                                            <input
                                                type="number"
                                                value={ragAgent.temperature}
                                                onChange={(e) => setRagAgent({ ...ragAgent, temperature: parseFloat(e.target.value) })}
                                                className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                min="0" max="2" step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-2">Retrieval K</label>
                                            <input
                                                type="number"
                                                value={ragAgent.retrieval_k}
                                                onChange={(e) => setRagAgent({ ...ragAgent, retrieval_k: parseInt(e.target.value) })}
                                                className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                min="1" max="20"
                                            />
                                        </div>
                                    </div>
                                </>
                            )}

                            {/* Translation-specific fields */}
                            {createAgentType === 'translation' && (
                                <>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-2">System prompt (optional)</label>
                                        <textarea
                                            value={translationAgent.system_prompt}
                                            onChange={(e) => setTranslationAgent({ ...translationAgent, system_prompt: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                            placeholder="Leave empty to use the default. Use {target_language} and {simplified} as placeholders."
                                            rows={3}
                                        />
                                    </div>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-2">Temperature</label>
                                            <input
                                                type="number"
                                                value={translationAgent.temperature}
                                                onChange={(e) => setTranslationAgent({ ...translationAgent, temperature: parseFloat(e.target.value) })}
                                                className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                min="0" max="2" step="0.1"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-slate-300 mb-2">Max Tokens</label>
                                            <input
                                                type="number"
                                                value={translationAgent.max_tokens}
                                                onChange={(e) => setTranslationAgent({ ...translationAgent, max_tokens: parseInt(e.target.value) })}
                                                className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                min="256" max="16384"
                                            />
                                        </div>
                                    </div>

                                    <div className="text-sm text-slate-500 bg-slate-800/30 rounded-lg p-3">
                                        <p>💡 Target language will be selected by users when translating.</p>
                                    </div>
                                </>
                            )}

                            <div className="flex gap-3 pt-4">
                                <button
                                    type="button"
                                    onClick={() => setShowCreateModal(false)}
                                    className="flex-1 px-4 py-3 bg-slate-800 text-slate-300 font-medium rounded-xl hover:bg-slate-700 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={
                                        createRAGMutation.isPending ||
                                        createTranslationMutation.isPending ||
                                        createVideoTranscriptionMutation.isPending ||
                                        (createAgentType === 'video_transcription'
                                            ? !videoTranscriptionConfig?.available
                                            : modelsLoading || !!modelsError || !modelsData?.models?.length)
                                    }
                                    className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-white font-semibold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed ${createAgentType === 'translation'
                                        ? 'bg-gradient-to-r from-emerald-600 to-teal-600'
                                        : createAgentType === 'video_transcription'
                                            ? 'bg-gradient-to-r from-blue-600 to-cyan-600'
                                            : 'bg-gradient-to-r from-violet-600 to-fuchsia-600'
                                        }`}
                                >
                                    {(createRAGMutation.isPending || createTranslationMutation.isPending || createVideoTranscriptionMutation.isPending) ? (
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                    ) : (
                                        'Create'
                                    )}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}

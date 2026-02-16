import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { agentsApi, dataDomainsApi } from '../api/client';
import { useAuth } from '../App';
import { ArrowLeft, MessageSquare, Loader2, Sparkles, Video, Pencil, Check, X } from 'lucide-react';
import VideoTranscription from '../components/VideoTranscription';

export default function AgentDetail() {
    const { id } = useParams<{ id: string }>();
    const { user } = useAuth();
    const queryClient = useQueryClient();
    const [editingPrompt, setEditingPrompt] = useState(false);
    const [promptDraft, setPromptDraft] = useState('');

    const { data: agent, isLoading } = useQuery({
        queryKey: ['agent', id],
        queryFn: () => agentsApi.get(parseInt(id!)).then(res => res.data),
        enabled: !!id,
    });

    const updateMutation = useMutation({
        mutationFn: (data: { system_prompt: string | null }) =>
            agentsApi.update(parseInt(id!), data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['agent', id] });
            setEditingPrompt(false);
        },
    });

    const { data: dataDomainsList } = useQuery({
        queryKey: ['dataDomains'],
        queryFn: () => dataDomainsApi.list().then(res => res.data),
        enabled: !!agent?.data_domain_ids?.length,
    });
    const agentDomains = (agent?.data_domain_ids?.length && dataDomainsList)
        ? dataDomainsList.filter((d) => agent!.data_domain_ids!.includes(d.id))
        : [];

    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
            </div>
        );
    }

    if (!agent) {
        return (
            <div className="text-center py-12">
                <p className="text-slate-400">Agent not found</p>
            </div>
        );
    }

    const canEdit = user && agent.owner_id === user.id;
    const showSystemPrompt =
        agent.agent_type === 'rag' || agent.agent_type === 'translation';

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            {/* Header */}
            <div>
                <Link
                    to="/agents"
                    className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-4 transition-colors"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Agents
                </Link>
                <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4">
                        <div className="flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20">
                            {agent.agent_type === 'video_transcription' ? (
                                <Video className="w-7 h-7 text-violet-400" />
                            ) : (
                                <Sparkles className="w-7 h-7 text-violet-400" />
                            )}
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-white">{agent.name}</h1>
                            <p className="text-slate-400">{agent.description || 'No description'}</p>
                        </div>
                    </div>
                    {agent.agent_type !== 'video_transcription' && (
                        <a
                            href={`http://localhost:7860?agent_id=${agent.id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all"
                        >
                            <MessageSquare className="w-5 h-5" />
                            Start Chat
                        </a>
                    )}
                </div>
            </div>

            {/* Video Transcription UI */}
            {agent.agent_type === 'video_transcription' ? (
                <VideoTranscription agentId={agent.id} />
            ) : (
                <>
                    {/* Configuration */}
                    <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-white mb-4">Configuration</h2>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                            <div>
                                <p className="text-sm text-slate-400 mb-1">LLM Model</p>
                                <p className="text-white font-medium">{agent.llm_model || 'N/A'}</p>
                            </div>
                            <div>
                                <p className="text-sm text-slate-400 mb-1">Temperature</p>
                                <p className="text-white font-medium">{agent.temperature || 'N/A'}</p>
                            </div>
                            <div>
                                <p className="text-sm text-slate-400 mb-1">Top P</p>
                                <p className="text-white font-medium">{agent.top_p || 'N/A'}</p>
                            </div>
                            <div>
                                <p className="text-sm text-slate-400 mb-1">Top K</p>
                                <p className="text-white font-medium">{agent.top_k || 'N/A'}</p>
                            </div>
                            <div>
                                <p className="text-sm text-slate-400 mb-1">Max Tokens</p>
                                <p className="text-white font-medium">{agent.max_tokens || 'N/A'}</p>
                            </div>
                            <div>
                                <p className="text-sm text-slate-400 mb-1">Retrieval K</p>
                                <p className="text-white font-medium">{agent.retrieval_k || 'N/A'}</p>
                            </div>
                            {agent.data_domain_ids?.length ? (
                                <div className="col-span-2">
                                    <p className="text-sm text-slate-400 mb-1">Data Domains</p>
                                    <div className="flex flex-wrap gap-2">
                                        {agentDomains.map((d) => (
                                            <Link
                                                key={d.id}
                                                to={`/data-domains/${d.id}`}
                                                className="text-violet-400 hover:text-violet-300 font-medium"
                                            >
                                                {d.name}
                                            </Link>
                                        ))}
                                        {agentDomains.length === 0 && agent.data_domain_ids?.length ? (
                                            <span className="text-slate-500">Loading...</span>
                                        ) : null}
                                    </div>
                                </div>
                            ) : null}
                        </div>
                    </div>

                    {/* System Prompt (RAG and Translation) */}
                    {showSystemPrompt && (
                        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h2 className="text-lg font-semibold text-white">System Prompt</h2>
                                {canEdit && !editingPrompt && (
                                    <button
                                        type="button"
                                        onClick={() => {
                                            setPromptDraft(agent.system_prompt || '');
                                            setEditingPrompt(true);
                                        }}
                                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-violet-400 transition-colors text-sm"
                                    >
                                        <Pencil className="w-4 h-4" />
                                        Edit
                                    </button>
                                )}
                            </div>
                            {editingPrompt ? (
                                <div className="space-y-3">
                                    <textarea
                                        value={promptDraft}
                                        onChange={(e) => setPromptDraft(e.target.value)}
                                        className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                        rows={6}
                                        placeholder={agent.agent_type === 'translation' ? 'Leave empty to use the default. Use {target_language} and {simplified} as placeholders.' : 'System prompt for the agent'}
                                    />
                                    <div className="flex gap-2">
                                        <button
                                            type="button"
                                            onClick={() => {
                                                updateMutation.mutate({ system_prompt: promptDraft.trim() || null });
                                            }}
                                            disabled={updateMutation.isPending}
                                            className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-medium disabled:opacity-50"
                                        >
                                            <Check className="w-4 h-4" />
                                            Save
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => { setEditingPrompt(false); setPromptDraft(''); }}
                                            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium"
                                        >
                                            <X className="w-4 h-4" />
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <pre className="text-slate-300 whitespace-pre-wrap font-mono text-sm bg-slate-800/50 rounded-xl p-4">
                                    {agent.system_prompt || (agent.agent_type === 'translation' ? 'No custom prompt (using built-in default).' : '')}
                                </pre>
                            )}
                        </div>
                    )}
                </>
            )}

            {/* Test Questions */}
            {agent.test_questions && agent.test_questions.length > 0 && (
                <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">Test Questions</h2>
                    <div className="space-y-4">
                        {agent.test_questions.map((test, idx) => (
                            <div key={idx} className="bg-slate-800/30 rounded-xl p-4">
                                <p className="text-sm text-slate-400 mb-1">Question {idx + 1}</p>
                                <p className="text-white mb-2">{test.question}</p>
                                <p className="text-sm text-slate-400 mb-1">Reference Answer</p>
                                <p className="text-slate-300 text-sm">{test.reference_answer}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}


import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { dataDomainsApi, modelsApi } from '../api/client';
import { Database, Plus, Trash2, FileText, Loader2, X, AlertCircle, RefreshCw, ChevronDown, ChevronUp, Video } from 'lucide-react';

export default function DataDomains() {
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [showVlmConfig, setShowVlmConfig] = useState(false);
    const [newDomain, setNewDomain] = useState({
        name: '',
        description: '',
        embedding_model: '',
        vlm_api_base: '',
        vlm_api_key: '',
        vlm_model_id: '',
        video_mode: 'procedure',
        vlm_prompt: '',
        object_tracker: 'yolo' as 'none' | 'simple_blob' | 'yolo' | 'yolo_api',
        enable_ocr: false,
    });
    const appliedVideoDefaultsRef = useRef(false);
    const queryClient = useQueryClient();

    // Fetch available embedding models
    const { data: modelsData, isLoading: modelsLoading, error: modelsError, refetch: refetchModels } = useQuery({
        queryKey: ['embeddingModels'],
        queryFn: () => modelsApi.listEmbedding().then(res => res.data),
    });

    // Fetch video ingestion defaults when create modal is open (VLM API + default prompts from env)
    const { data: videoDefaults } = useQuery({
        queryKey: ['videoIngestionDefaults'],
        queryFn: () => dataDomainsApi.getVideoDefaults(),
        enabled: showCreateModal,
    });

    // Apply video defaults to form when modal opens and defaults are loaded
    useEffect(() => {
        if (!showCreateModal) {
            appliedVideoDefaultsRef.current = false;
            return;
        }
        if (!videoDefaults || appliedVideoDefaultsRef.current) return;
        appliedVideoDefaultsRef.current = true;
        setNewDomain((prev) => {
            const defaultPrompt =
                videoDefaults.prompt_library?.[prev.video_mode]
                ?? (prev.video_mode === 'race' ? videoDefaults.default_prompt_race : videoDefaults.default_prompt_procedure)
                ?? '';
            return {
                ...prev,
                vlm_api_base: videoDefaults.vlm_api_base ?? prev.vlm_api_base,
                vlm_api_key: videoDefaults.vlm_api_key ?? prev.vlm_api_key,
                vlm_model_id: videoDefaults.vlm_model ?? prev.vlm_model_id,
                vlm_prompt: defaultPrompt,
            };
        });
    }, [showCreateModal, videoDefaults]);

    // Set default model when models are loaded
    useEffect(() => {
        if (modelsData?.models && modelsData.models.length > 0 && !newDomain.embedding_model) {
            setNewDomain(prev => ({ ...prev, embedding_model: modelsData.models[0].id }));
        }
    }, [modelsData]);

    const { data: domains, isLoading } = useQuery({
        queryKey: ['dataDomains'],
        queryFn: () => dataDomainsApi.list().then(res => res.data),
    });

    const createMutation = useMutation({
        mutationFn: (data: typeof newDomain) => {
            const fromLibrary = videoDefaults?.prompt_library?.[data.video_mode];
            const fallback = data.video_mode === 'race' ? videoDefaults?.default_prompt_race : videoDefaults?.default_prompt_procedure;
            const vlm_prompt = data.vlm_prompt || fromLibrary || fallback || null;
            return dataDomainsApi.create({
                name: data.name,
                description: data.description || undefined,
                embedding_model: data.embedding_model,
                vlm_api_base: data.vlm_api_base || null,
                vlm_api_key: data.vlm_api_key || null,
                vlm_model_id: data.vlm_model_id || null,
                video_mode: data.video_mode || null,
                vlm_prompt,
                object_tracker: data.object_tracker || 'yolo',
                enable_ocr: data.enable_ocr ?? false,
            });
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['dataDomains'] });
            setShowCreateModal(false);
            setShowVlmConfig(false);
            appliedVideoDefaultsRef.current = false;
            setNewDomain({
                name: '',
                description: '',
                embedding_model: modelsData?.models[0]?.id || '',
                vlm_api_base: '',
                vlm_api_key: '',
                vlm_model_id: '',
                video_mode: 'procedure',
                vlm_prompt: '',
                object_tracker: 'yolo',
                enable_ocr: false,
            });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: (id: number) => dataDomainsApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['dataDomains'] });
        },
    });

    const handleCreate = (e: React.FormEvent) => {
        e.preventDefault();
        createMutation.mutate(newDomain);
    };

    return (
        <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Data Domains</h1>
                    <p className="text-slate-400">Manage your document collections</p>
                </div>
                <button
                    onClick={() => setShowCreateModal(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all"
                >
                    <Plus className="w-5 h-5" />
                    Create Domain
                </button>
            </div>

            {/* Loading */}
            {isLoading && (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
                </div>
            )}

            {/* Domains grid */}
            {domains && domains.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {domains.map((domain) => (
                        <Link
                            key={domain.id}
                            to={`/data-domains/${domain.id}`}
                            className="group bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6 hover:border-violet-500/30 transition-all"
                        >
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-violet-500/10">
                                    <Database className="w-6 h-6 text-violet-400" />
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.preventDefault();
                                        if (confirm('Delete this data domain?')) {
                                            deleteMutation.mutate(domain.id);
                                        }
                                    }}
                                    className="p-2 rounded-lg text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                                >
                                    <Trash2 className="w-5 h-5" />
                                </button>
                            </div>
                            <h3 className="text-lg font-semibold text-white mb-1">{domain.name}</h3>
                            <p className="text-slate-400 text-sm mb-4 line-clamp-2">
                                {domain.description || 'No description'}
                            </p>
                            <div className="flex items-center gap-2 text-sm text-slate-500">
                                <FileText className="w-4 h-4" />
                                <span>{domain.document_count || 0} documents</span>
                            </div>
                            <div className="mt-2 text-xs text-slate-600 truncate">
                                Model: {domain.embedding_model}
                            </div>
                        </Link>
                    ))}
                </div>
            )}

            {/* Empty state */}
            {domains && domains.length === 0 && (
                <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-12 text-center">
                    <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-2xl bg-violet-500/10">
                        <Database className="w-8 h-8 text-violet-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">No data domains yet</h3>
                    <p className="text-slate-400 mb-6">
                        Create your first data domain to start uploading documents.
                    </p>
                    <button
                        onClick={() => setShowCreateModal(true)}
                        className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all"
                    >
                        <Plus className="w-5 h-5" />
                        Create Data Domain
                    </button>
                </div>
            )}

            {/* Create Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md max-h-[90vh] flex flex-col shadow-xl">
                        <div className="flex items-center justify-between p-6 pb-0 flex-shrink-0">
                            <h2 className="text-xl font-semibold text-white">Create Data Domain</h2>
                            <button
                                onClick={() => setShowCreateModal(false)}
                                className="p-2 rounded-lg text-slate-400 hover:bg-slate-800"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <form onSubmit={handleCreate} className="flex flex-col flex-1 min-h-0 overflow-hidden">
                            <div className="flex-1 min-h-0 overflow-y-auto px-6 py-4 space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">Name</label>
                                    <input
                                        type="text"
                                        value={newDomain.name}
                                        onChange={(e) => setNewDomain({ ...newDomain, name: e.target.value })}
                                        className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                        placeholder="My Documents"
                                        required
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Description (optional)
                                    </label>
                                    <textarea
                                        value={newDomain.description}
                                        onChange={(e) => setNewDomain({ ...newDomain, description: e.target.value })}
                                        className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                        placeholder="A collection of..."
                                        rows={3}
                                    />
                                </div>

                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <label className="block text-sm font-medium text-slate-300">
                                            Embedding Model
                                        </label>
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
                                        <div className="flex items-center gap-2 text-amber-400 text-sm py-3 bg-amber-500/10 px-3 rounded-lg">
                                            <AlertCircle className="w-4 h-4" />
                                            <span>Could not load models from endpoint</span>
                                            <button
                                                type="button"
                                                onClick={() => refetchModels()}
                                                className="ml-auto text-amber-300 hover:text-amber-200"
                                            >
                                                Retry
                                            </button>
                                        </div>
                                    )}

                                    {modelsData && modelsData.models.length === 0 && (
                                        <div className="flex items-center gap-2 text-amber-400 text-sm py-3 bg-amber-500/10 px-3 rounded-lg">
                                            <AlertCircle className="w-4 h-4" />
                                            <span>No embedding models available at {modelsData.endpoint}</span>
                                        </div>
                                    )}

                                    {modelsData && modelsData.models.length > 0 && (
                                        <>
                                            <select
                                                value={newDomain.embedding_model}
                                                onChange={(e) => setNewDomain({ ...newDomain, embedding_model: e.target.value })}
                                                className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                required
                                            >
                                                {modelsData.models.map((model) => (
                                                    <option key={model.id} value={model.id}>
                                                        {model.name}
                                                    </option>
                                                ))}
                                            </select>
                                            <p className="mt-1 text-xs text-slate-500">
                                                From: {modelsData.endpoint}
                                            </p>
                                        </>
                                    )}
                                </div>

                                {/* Video / VLM configuration: optional; only "active" when VLM_API_BASE is set in .env */}
                                <div className="border-t border-slate-800/50 pt-4">
                                    <button
                                        type="button"
                                        onClick={() => setShowVlmConfig(!showVlmConfig)}
                                        className="flex items-center justify-between w-full text-left text-sm font-medium text-slate-300 hover:text-white transition-colors"
                                    >
                                        <div className="flex items-center gap-2">
                                            <Video className="w-4 h-4" />
                                            <span>
                                                Video Processing (Optional)
                                                {videoDefaults?.video_ingestion_available ? (
                                                    <span className="ml-2 text-violet-400 font-normal">— VLM configured</span>
                                                ) : (
                                                    <span className="ml-2 text-slate-500 font-normal">— Set VLM_API_BASE in .env to enable</span>
                                                )}
                                            </span>
                                        </div>
                                        {showVlmConfig ? (
                                            <ChevronUp className="w-4 h-4" />
                                        ) : (
                                            <ChevronDown className="w-4 h-4" />
                                        )}
                                    </button>

                                    {showVlmConfig && (
                                        <div className="mt-4 space-y-4">
                                            <div>
                                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                                    VLM API Base URL
                                                </label>
                                                <input
                                                    type="text"
                                                    value={newDomain.vlm_api_base}
                                                    onChange={(e) => setNewDomain({ ...newDomain, vlm_api_base: e.target.value })}
                                                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                    placeholder="https://api.cosmos.example/v1"
                                                />
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                                    VLM API Key
                                                </label>
                                                <input
                                                    type="password"
                                                    value={newDomain.vlm_api_key}
                                                    onChange={(e) => setNewDomain({ ...newDomain, vlm_api_key: e.target.value })}
                                                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                    placeholder="Your VLM API key"
                                                />
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                                    VLM Model ID
                                                </label>
                                                <input
                                                    type="text"
                                                    value={newDomain.vlm_model_id}
                                                    onChange={(e) => setNewDomain({ ...newDomain, vlm_model_id: e.target.value })}
                                                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                    placeholder="cosmos-vision-1"
                                                />
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                                    Video Prompt
                                                </label>
                                                <select
                                                    value={newDomain.video_mode}
                                                    onChange={(e) => {
                                                        const mode = e.target.value;
                                                        const defaultPrompt =
                                                            videoDefaults?.prompt_library?.[mode]
                                                            ?? (mode === 'race' ? videoDefaults?.default_prompt_race : videoDefaults?.default_prompt_procedure)
                                                            ?? '';
                                                        setNewDomain({ ...newDomain, video_mode: mode, vlm_prompt: defaultPrompt });
                                                    }}
                                                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                >
                                                    <option value="procedure">Procedure (steps, actions, tools)</option>
                                                    <option value="race">Race (per-track commentary, interactions)</option>
                                                </select>
                                                <p className="mt-1 text-xs text-slate-500">
                                                    The prompt is editable on the domain page after creation.
                                                </p>
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                                    Object Tracker
                                                </label>
                                                <select
                                                    value={newDomain.object_tracker}
                                                    onChange={(e) => setNewDomain({ ...newDomain, object_tracker: e.target.value as 'none' | 'simple_blob' | 'yolo' | 'yolo_api' })}
                                                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                                >
                                                    <option value="yolo">YOLO (local)</option>
                                                    <option value="yolo_api">YOLO API (Roboflow)</option>
                                                    <option value="simple_blob">Simple Blob (motion detection)</option>
                                                    <option value="none">None</option>
                                                </select>
                                                <p className="mt-1 text-xs text-slate-500">
                                                    YOLO detects objects with class labels. Use for race/traffic videos.
                                                </p>
                                            </div>

                                            <div className="flex items-center gap-3">
                                                <input
                                                    type="checkbox"
                                                    id="enable_ocr"
                                                    checked={newDomain.enable_ocr}
                                                    onChange={(e) => setNewDomain({ ...newDomain, enable_ocr: e.target.checked })}
                                                    className="w-4 h-4 rounded bg-slate-800 border-slate-600 text-violet-500 focus:ring-violet-500"
                                                />
                                                <label htmlFor="enable_ocr" className="text-sm text-slate-300">
                                                    Enable OCR on video frames
                                                </label>
                                            </div>
                                            <p className="text-xs text-slate-500 -mt-2">
                                                Extract text from each frame using Tesseract.
                                            </p>
                                        </div>
                                    )}
                                </div>

                            </div>

                            <div className="flex gap-3 p-6 pt-4 border-t border-slate-800/50 flex-shrink-0 bg-slate-900 rounded-b-2xl">
                                <button
                                    type="button"
                                    onClick={() => setShowCreateModal(false)}
                                    className="flex-1 px-4 py-3 bg-slate-800 text-slate-300 font-medium rounded-xl hover:bg-slate-700 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={createMutation.isPending || !newDomain.embedding_model}
                                    className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white font-semibold rounded-xl disabled:opacity-50"
                                >
                                    {createMutation.isPending ? (
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

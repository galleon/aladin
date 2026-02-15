import { useCallback, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useDropzone } from 'react-dropzone';
import { dataDomainsApi } from '../api/client';
import ErrorModal from '../components/ErrorModal';
import ReactMarkdown from 'react-markdown';
import {
    ArrowLeft,
    Upload,
    FileText,
    Trash2,
    Loader2,
    CheckCircle,
    XCircle,
    Clock,
    Video,
    Filter,
    Search,
    RefreshCw,
    Pencil,
    Save,
    X,
} from 'lucide-react';

export default function DataDomainDetail() {
    const { id } = useParams<{ id: string }>();
    const queryClient = useQueryClient();
    const [fileTypeFilter, setFileTypeFilter] = useState<'all' | 'document' | 'video'>('all');
    const [editingPrompt, setEditingPrompt] = useState(false);
    const [promptEditValue, setPromptEditValue] = useState('');
    const [errorModal, setErrorModal] = useState<{ isOpen: boolean; errorMessage: string }>({
        isOpen: false,
        errorMessage: '',
    });

    const { data: domain, isLoading } = useQuery({
        queryKey: ['dataDomain', id],
        queryFn: () => dataDomainsApi.get(parseInt(id!)).then(res => res.data),
        enabled: !!id,
        refetchInterval: 5000, // Refresh to update document status
    });

    const uploadMutation = useMutation({
        mutationFn: (file: File) => dataDomainsApi.uploadDocument(parseInt(id!), file),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['dataDomain', id] });
            queryClient.invalidateQueries({ queryKey: ['unifiedJobs'] });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: (documentId: number) => dataDomainsApi.deleteDocument(parseInt(id!), documentId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['dataDomain', id] });
        },
    });

    const reindexDomainMutation = useMutation({
        mutationFn: () => dataDomainsApi.reindexDomain(parseInt(id!)),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['dataDomain', id] });
            queryClient.invalidateQueries({ queryKey: ['unifiedJobs'] });
        },
    });

    const reindexDocumentMutation = useMutation({
        mutationFn: (documentId: number) =>
            dataDomainsApi.reindexDocument(parseInt(id!), documentId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['dataDomain', id] });
            queryClient.invalidateQueries({ queryKey: ['unifiedJobs'] });
        },
    });

    const updateDomainMutation = useMutation({
        mutationFn: (data: { vlm_prompt?: string | null }) =>
            dataDomainsApi.update(parseInt(id!), data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['dataDomain', id] });
            setEditingPrompt(false);
        },
    });

    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            acceptedFiles.forEach((file) => {
                uploadMutation.mutate(file);
            });
        },
        [uploadMutation]
    );

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'text/plain': ['.txt'],
            'text/markdown': ['.md'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
            'text/csv': ['.csv'],
            'application/json': ['.json'],
            'video/mp4': ['.mp4'],
        },
    });

    // Filter documents by type
    const filteredDocuments = useMemo(() => {
        if (!domain?.documents) return [];
        if (fileTypeFilter === 'all') return domain.documents;
        if (fileTypeFilter === 'video') {
            return domain.documents.filter(doc => doc.file_type === 'mp4' || doc.processing_type === 'video');
        }
        return domain.documents.filter(doc => doc.file_type !== 'mp4' && doc.processing_type !== 'video');
    }, [domain?.documents, fileTypeFilter]);

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'ready':
                return <CheckCircle className="w-5 h-5 text-green-400" />;
            case 'failed':
                return <XCircle className="w-5 h-5 text-red-400" />;
            case 'processing':
                return <Loader2 className="w-5 h-5 text-violet-400 animate-spin" />;
            default:
                return <Clock className="w-5 h-5 text-slate-400" />;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'ready':
                return 'bg-green-500/10 text-green-400';
            case 'failed':
                return 'bg-red-500/10 text-red-400';
            case 'processing':
                return 'bg-violet-500/10 text-violet-400';
            default:
                return 'bg-slate-500/10 text-slate-400';
        }
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
            </div>
        );
    }

    if (!domain) {
        return (
            <div className="text-center py-12">
                <p className="text-slate-400">Data domain not found</p>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            {/* Header */}
            <div>
                <Link
                    to="/data-domains"
                    className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-4 transition-colors"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Data Domains
                </Link>
                <h1 className="text-3xl font-bold text-white mb-2">{domain.name}</h1>
                <p className="text-slate-400">{domain.description || 'No description'}</p>
            </div>

            {/* Info */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-6 mb-4">
                    <div>
                        <p className="text-sm text-slate-400 mb-1">Embedding Model</p>
                        <p className="text-white font-medium">{domain.embedding_model}</p>
                    </div>
                    <div>
                        <p className="text-sm text-slate-400 mb-1">Collection</p>
                        <p className="text-white font-medium font-mono text-sm">{domain.qdrant_collection}</p>
                    </div>
                    <div>
                        <p className="text-sm text-slate-400 mb-1">Documents</p>
                        <p className="text-white font-medium">{domain.documents?.length || 0}</p>
                    </div>
                </div>
                <div className="pt-4 border-t border-slate-800/50 flex flex-wrap items-center gap-3">
                    <Link
                        to={`/data-domains/${id}/inspect`}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800/50 text-slate-300 hover:bg-violet-500/20 hover:text-violet-300 border border-slate-700/50 hover:border-violet-500/30 transition-colors text-sm"
                    >
                        <Search className="w-4 h-4" />
                        Inspect collection (view stored chunks)
                    </Link>
                    <button
                        type="button"
                        onClick={() => {
                            if (
                                confirm(
                                    'Reindex the entire domain? This will clear the vector store and re-ingest all documents. Continue?'
                                )
                            ) {
                                reindexDomainMutation.mutate();
                            }
                        }}
                        disabled={reindexDomainMutation.isPending || !domain.documents?.length}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 border border-amber-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
                    >
                        {reindexDomainMutation.isPending ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <RefreshCw className="w-4 h-4" />
                        )}
                        Reindex all
                    </button>
                </div>
                {(domain.vlm_api_base || domain.vlm_model_id || domain.object_tracker || domain.enable_ocr) && (
                    <div className="border-t border-slate-800/50 pt-4 mt-4">
                        <p className="text-sm text-slate-400 mb-2">Video Processing (VLM)</p>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            {domain.vlm_api_base && (
                                <div>
                                    <p className="text-slate-500 text-xs mb-1">API Base</p>
                                    <p className="text-white font-mono text-xs truncate">{domain.vlm_api_base}</p>
                                </div>
                            )}
                            {domain.vlm_model_id && (
                                <div>
                                    <p className="text-slate-500 text-xs mb-1">Model</p>
                                    <p className="text-white">{domain.vlm_model_id}</p>
                                </div>
                            )}
                            {domain.video_mode && (
                                <div>
                                    <p className="text-slate-500 text-xs mb-1">Mode</p>
                                    <p className="text-white capitalize">{domain.video_mode}</p>
                                </div>
                            )}
                            {domain.object_tracker && domain.object_tracker !== 'none' && (
                                <div>
                                    <p className="text-slate-500 text-xs mb-1">Object Tracker</p>
                                    <p className="text-white capitalize">{domain.object_tracker.replace('_', ' ')}</p>
                                </div>
                            )}
                            {domain.enable_ocr && (
                                <div>
                                    <p className="text-slate-500 text-xs mb-1">OCR</p>
                                    <p className="text-white">Enabled</p>
                                </div>
                            )}
                        </div>
                        <div className="mt-3">
                            <div className="flex items-center justify-between mb-1">
                                <p className="text-slate-500 text-xs">Custom Prompt</p>
                                {!editingPrompt ? (
                                    <button
                                        type="button"
                                        onClick={() => {
                                            setPromptEditValue(domain.vlm_prompt ?? '');
                                            setEditingPrompt(true);
                                        }}
                                        className="text-violet-400 hover:text-violet-300 text-xs flex items-center gap-1"
                                    >
                                        <Pencil className="w-3.5 h-3.5" />
                                        Edit
                                    </button>
                                ) : null}
                            </div>
                            {editingPrompt ? (
                                <>
                                    <textarea
                                        value={promptEditValue}
                                        onChange={(e) => setPromptEditValue(e.target.value)}
                                        className="w-full px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white text-xs font-mono placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 min-h-[120px]"
                                        placeholder="Optional: custom VLM prompt. Placeholders: {t_start}, {t_end}, {num_frames}, {tracks_json}"
                                    />
                                    <div className="flex gap-2 mt-2">
                                        <button
                                            type="button"
                                            onClick={() => {
                                                updateDomainMutation.mutate({ vlm_prompt: promptEditValue.trim() || null });
                                            }}
                                            disabled={updateDomainMutation.isPending}
                                            className="flex items-center gap-1 px-3 py-1.5 bg-violet-600 hover:bg-violet-500 text-white text-xs rounded-lg disabled:opacity-50"
                                        >
                                            {updateDomainMutation.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                                            Save
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setEditingPrompt(false);
                                                setPromptEditValue('');
                                            }}
                                            className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded-lg"
                                        >
                                            <X className="w-3.5 h-3.5" />
                                            Cancel
                                        </button>
                                    </div>
                                </>
                            ) : (
                                <div className="text-white text-xs max-h-48 overflow-y-auto [&_strong]:text-violet-300 [&_code]:text-violet-200 [&_code]:bg-slate-800 [&_code]:px-1 [&_code]:rounded [&_p]:my-1 [&_ul]:my-1 [&_li]:my-0 [&_ul]:list-disc [&_ul]:pl-4">
                                    {domain.vlm_prompt ? (
                                        <ReactMarkdown
                                            components={{
                                                p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
                                                strong: ({ children }) => <strong className="text-violet-300 font-semibold">{children}</strong>,
                                                ul: ({ children }) => <ul className="list-disc pl-4 my-1">{children}</ul>,
                                                li: ({ children }) => <li className="my-0">{children}</li>,
                                                code: ({ children }) => <code className="text-violet-200 bg-slate-800 px-1 rounded text-[0.7rem]">{children}</code>,
                                            }}
                                        >
                                            {domain.vlm_prompt}
                                        </ReactMarkdown>
                                    ) : (
                                        <p className="text-slate-500 italic">No custom prompt set. Click Edit to add one.</p>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Upload area */}
            <div
                {...getRootProps()}
                className={`bg-slate-900/50 backdrop-blur-sm border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${isDragActive
                        ? 'border-violet-500 bg-violet-500/10'
                        : 'border-slate-700/50 hover:border-violet-500/50'
                    }`}
            >
                <input {...getInputProps()} />
                <div className="flex items-center justify-center w-14 h-14 mx-auto mb-4 rounded-xl bg-violet-500/10">
                    <Upload className="w-7 h-7 text-violet-400" />
                </div>
                {uploadMutation.isPending ? (
                    <div className="flex items-center justify-center gap-2">
                        <Loader2 className="w-5 h-5 text-violet-400 animate-spin" />
                        <span className="text-slate-300">Uploading...</span>
                    </div>
                ) : isDragActive ? (
                    <p className="text-violet-300">Drop files here...</p>
                ) : (
                    <>
                        <p className="text-white font-medium mb-1">
                            Drag and drop files here, or click to select
                        </p>
                        <p className="text-slate-400 text-sm">
                            Supports PDF, TXT, MD, DOCX, CSV, JSON, MP4 (max 50MB for documents, 500MB for videos)
                        </p>
                    </>
                )}
            </div>

            {/* Documents list */}
            {domain.documents && domain.documents.length > 0 && (
                <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl overflow-hidden">
                    <div className="p-4 border-b border-slate-800/50 flex items-center justify-between">
                        <h2 className="text-lg font-semibold text-white">Documents</h2>
                        <div className="flex items-center gap-2">
                            <Filter className="w-4 h-4 text-slate-400" />
                            <select
                                value={fileTypeFilter}
                                onChange={(e) => setFileTypeFilter(e.target.value as 'all' | 'document' | 'video')}
                                className="px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                            >
                                <option value="all">All Files</option>
                                <option value="document">Documents</option>
                                <option value="video">Videos</option>
                            </select>
                        </div>
                    </div>
                    <div className="divide-y divide-slate-800/50">
                        {filteredDocuments.length === 0 ? (
                            <div className="p-8 text-center text-slate-400">
                                No {fileTypeFilter === 'all' ? '' : fileTypeFilter} files found
                            </div>
                        ) : (
                            filteredDocuments.map((doc) => (
                                <div key={doc.id} className="p-4 flex items-center gap-4">
                                    <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-slate-800/50">
                                        {(doc.file_type === 'mp4' || doc.processing_type === 'video') ? (
                                            <Video className="w-5 h-5 text-violet-400" />
                                        ) : (
                                            <FileText className="w-5 h-5 text-slate-400" />
                                        )}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="text-white font-medium truncate">{doc.original_filename}</p>
                                        <div className="flex items-center gap-3 text-sm text-slate-400">
                                            <span>{doc.file_type?.toUpperCase()}</span>
                                            {doc.file_size && (
                                                <span>{(doc.file_size / 1024).toFixed(1)} KB</span>
                                            )}
                                            {doc.chunk_count > 0 && <span>{doc.chunk_count} chunks</span>}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        {doc.status === 'failed' && doc.error_message ? (
                                            <button
                                                onClick={() =>
                                                    setErrorModal({ isOpen: true, errorMessage: doc.error_message || '' })
                                                }
                                                className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm cursor-pointer hover:opacity-80 transition-opacity ${getStatusColor(
                                                    doc.status
                                                )}`}
                                                title="Click to view error details"
                                            >
                                                {getStatusIcon(doc.status)}
                                                {doc.status}
                                            </button>
                                        ) : doc.status === 'failed' ? (
                                            <span
                                                className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm cursor-default ${getStatusColor(
                                                    doc.status
                                                )}`}
                                                title="No error message available"
                                            >
                                                {getStatusIcon(doc.status)}
                                                {doc.status}
                                            </span>
                                        ) : (
                                            <span
                                                className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm cursor-default ${getStatusColor(
                                                    doc.status
                                                )}`}
                                            >
                                                {getStatusIcon(doc.status)}
                                                {doc.status}
                                            </span>
                                        )}
                                        <button
                                            onClick={() => {
                                                if (
                                                    confirm(
                                                        'Reindex this document? Its chunks will be removed and the file re-ingested.'
                                                    )
                                                ) {
                                                    reindexDocumentMutation.mutate(doc.id);
                                                }
                                            }}
                                            disabled={reindexDocumentMutation.isPending}
                                            className="p-2 rounded-lg text-slate-500 hover:bg-amber-500/10 hover:text-amber-400 transition-colors disabled:opacity-50"
                                            title="Reindex (re-ingest)"
                                        >
                                            <RefreshCw className="w-5 h-5" />
                                        </button>
                                        <button
                                            onClick={() => {
                                                if (confirm('Delete this document?')) {
                                                    deleteMutation.mutate(doc.id);
                                                }
                                            }}
                                            className="p-2 rounded-lg text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-colors"
                                        >
                                            <Trash2 className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}

            <ErrorModal
                isOpen={errorModal.isOpen}
                onClose={() => setErrorModal({ isOpen: false, errorMessage: '' })}
                errorMessage={errorModal.errorMessage}
                title="Document Processing Error"
            />
        </div>
    );
}


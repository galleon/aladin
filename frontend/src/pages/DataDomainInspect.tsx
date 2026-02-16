import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { dataDomainsApi } from '../api/client';
import { ArrowLeft, Loader2, FileText, Video, Copy, ChevronDown, ChevronUp } from 'lucide-react';

const PREVIEW_LENGTH = 180;
const METADATA_KEYS_OMIT = ['content', 'text', 'frame_times', 'fields'];

function formatTime(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function getPreviewText(payload: Record<string, unknown>): string {
    const text = (payload.content ?? payload.text) as string | undefined;
    return typeof text === 'string' ? text : '';
}

function isVideoChunk(payload: Record<string, unknown>): boolean {
    return 't_start' in payload && 't_end' in payload;
}

function getMetadataEntries(payload: Record<string, unknown>): [string, unknown][] {
    return Object.entries(payload).filter(
        ([k]) => !METADATA_KEYS_OMIT.includes(k)
    );
}

const CHUNKS_PAGE_SIZE = 20;

export default function DataDomainInspect() {
    const { id } = useParams<{ id: string }>();
    const domainId = id ? parseInt(id, 10) : NaN;
    const [items, setItems] = useState<{ id: string; payload: Record<string, unknown> }[]>([]);
    const [hasMore, setHasMore] = useState(false);
    const [loading, setLoading] = useState(false);
    const [loadingMore, setLoadingMore] = useState(false);
    const [expandedId, setExpandedId] = useState<string | null>(null);

    const { data: domain, isLoading: domainLoading } = useQuery({
        queryKey: ['dataDomain', id],
        queryFn: () => dataDomainsApi.get(domainId).then(res => res.data),
        enabled: !!id && !isNaN(domainId),
    });

    const fetchChunks = useCallback(
        async (offset: number, append: boolean) => {
            if (isNaN(domainId)) return;
            const setLoader = append ? setLoadingMore : setLoading;
            setLoader(true);
            try {
                const res = await dataDomainsApi.getChunks(domainId, {
                    limit: CHUNKS_PAGE_SIZE,
                    offset,
                });
                const data = res.data;
                if (append) {
                    setItems(prev => [...prev, ...data.items]);
                } else {
                    setItems(data.items);
                }
                setHasMore(data.has_more);
            } finally {
                setLoader(false);
            }
        },
        [domainId]
    );

    useEffect(() => {
        if (!domainId || !domain || domain.id !== domainId) return;
        fetchChunks(0, false);
    }, [domainId, domain?.id, fetchChunks]);

    const loadMore = () => {
        if (hasMore) fetchChunks(items.length, true);
    };

    const copyId = (pointId: string) => {
        navigator.clipboard.writeText(pointId);
    };

    if (domainLoading || !domain) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <div>
                <Link
                    to={`/data-domains/${domainId}`}
                    className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-4 transition-colors"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to {domain.name}
                </Link>
                <h1 className="text-3xl font-bold text-white mb-2">Inspect collection</h1>
                <p className="text-slate-400">
                    Stored chunks in vector store for <span className="font-mono text-slate-300">{domain.qdrant_collection}</span>.
                    List is ordered by source file and chunk index.
                </p>
            </div>

            {loading ? (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
                </div>
            ) : items.length === 0 ? (
                <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-12 text-center text-slate-400">
                    No chunks in this collection yet. Ingest documents or videos from the data domain page.
                </div>
            ) : (
                <div className="space-y-4">
                    {(() => {
                        let lastSourceFile: string | undefined;
                        return items.map((item) => {
                            const sourceFile = (item.payload.source_file as string) ?? '';
                            const showDocHeader = sourceFile !== lastSourceFile;
                            if (showDocHeader) lastSourceFile = sourceFile;

                            const text = getPreviewText(item.payload);
                            const isVideo = isVideoChunk(item.payload);
                            const isExpanded = expandedId === item.id;
                            const tStart = item.payload.t_start as number | undefined;
                            const tEnd = item.payload.t_end as number | undefined;
                            const timeRange =
                                typeof tStart === 'number' && typeof tEnd === 'number'
                                    ? `${formatTime(tStart)} – ${formatTime(tEnd)}`
                                    : null;
                            const preview =
                                text.length <= PREVIEW_LENGTH
                                    ? text
                                    : text.slice(0, PREVIEW_LENGTH) + '…';
                            const metadataEntries = getMetadataEntries(item.payload);

                            return (
                                <div key={item.id} className="space-y-2">
                                    {showDocHeader && sourceFile ? (
                                        <div className="flex items-center gap-2 py-2 border-b border-slate-700/50">
                                            <FileText className="w-4 h-4 text-slate-500 flex-shrink-0" />
                                            <span className="text-sm font-medium text-slate-400 truncate" title={sourceFile}>
                                                Document: {sourceFile}
                                            </span>
                                        </div>
                                    ) : null}
                                    <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl overflow-hidden">
                                <div className="p-4 flex flex-col gap-3">
                                    <div className="flex items-center justify-between gap-2 flex-wrap">
                                        <div className="flex items-center gap-2 min-w-0">
                                            <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-800/50 flex items-center justify-center">
                                                {isVideo ? (
                                                    <Video className="w-4 h-4 text-violet-400" />
                                                ) : (
                                                    <FileText className="w-4 h-4 text-slate-400" />
                                                )}
                                            </div>
                                            <div className="min-w-0 flex-1">
                                                <div className="flex items-center gap-2 flex-wrap">
                                                    <code className="text-xs text-slate-400 font-mono truncate max-w-[180px]" title={item.id}>
                                                        {item.id.slice(0, 8)}…
                                                    </code>
                                                    <button
                                                        type="button"
                                                        onClick={() => copyId(item.id)}
                                                        className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-700/50"
                                                        title="Copy ID"
                                                    >
                                                        <Copy className="w-3.5 h-3.5" />
                                                    </button>
                                                </div>
                                                {timeRange && (
                                                    <p className="text-xs text-violet-400 mt-0.5">{timeRange}</p>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="text-sm text-slate-300 whitespace-pre-wrap break-words">
                                        {isExpanded ? text : preview}
                                        {text.length > PREVIEW_LENGTH && (
                                            <button
                                                type="button"
                                                onClick={() => setExpandedId(isExpanded ? null : item.id)}
                                                className="ml-2 text-violet-400 hover:underline inline-flex items-center gap-0.5"
                                            >
                                                {isExpanded ? (
                                                    <>Show less <ChevronUp className="w-4 h-4" /></>
                                                ) : (
                                                    <>Show more <ChevronDown className="w-4 h-4" /></>
                                                )}
                                            </button>
                                        )}
                                    </div>

                                    {metadataEntries.length > 0 && (
                                        <div className="border-t border-slate-800/50 pt-3 mt-1">
                                            <p className="text-xs text-slate-500 mb-2">Metadata</p>
                                            <div className="flex flex-wrap gap-2">
                                                {metadataEntries.map(([key, value]) => {
                                                    const display =
                                                        Array.isArray(value)
                                                            ? `[${value.length} items]`
                                                            : typeof value === 'object' && value !== null
                                                                ? JSON.stringify(value).slice(0, 50) + '…'
                                                                : String(value);
                                                    return (
                                                        <span
                                                            key={key}
                                                            className="px-2 py-1 rounded-lg bg-slate-800/50 text-xs text-slate-400"
                                                            title={typeof value === 'string' ? value : JSON.stringify(value)}
                                                        >
                                                            <span className="text-slate-500">{key}:</span> {display}
                                                        </span>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </div>
                                    </div>
                                </div>
                            );
                        });
                    })()}

                    {hasMore && (
                        <div className="flex justify-center pt-4">
                            <button
                                type="button"
                                onClick={loadMore}
                                disabled={loadingMore}
                                className="px-6 py-2.5 rounded-xl bg-violet-500/20 text-violet-300 hover:bg-violet-500/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                {loadingMore ? (
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                ) : null}
                                Load more
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

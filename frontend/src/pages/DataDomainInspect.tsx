import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { dataDomainsApi, clipsApi } from '../api/client';
import { ArrowLeft, Loader2, FileText, Video, Copy, ChevronDown, ChevronUp, Table2, Image, Play } from 'lucide-react';

const PREVIEW_LENGTH = 180;
// Keys rendered with dedicated UI — excluded from the flat metadata pill list
const METADATA_KEYS_OMIT = ['content', 'text', 'frame_times', 'fields', 'table_content', 'image_data', 'image_caption', 'cv_meta'];

/** Render a GFM markdown table string as an HTML <table>. Falls back to a <pre> block for non-table input. */
function MarkdownTable({ md }: { md: string }) {
    const lines = md.trim().split('\n').filter(l => l.trim().startsWith('|'));
    if (lines.length < 2) return <pre className="text-xs text-slate-400 whitespace-pre-wrap">{md}</pre>;
    const parseRow = (line: string) =>
        line.split('|').slice(1, -1).map(c => c.trim());
    const headers = parseRow(lines[0]);
    const body = lines.slice(2).map(parseRow); // skip separator row
    return (
        <table className="w-full text-xs border-collapse">
            <thead>
                <tr>
                    {headers.map((h, i) => (
                        <th key={i} className="bg-slate-800/60 text-slate-300 px-2 py-1 border border-slate-700/50 text-left font-medium">
                            {h}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {body.map((row, ri) => (
                    <tr key={ri} className="even:bg-slate-800/20">
                        {row.map((cell, ci) => (
                            <td key={ci} className="text-slate-400 px-2 py-1 border border-slate-700/50">
                                {cell}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

const CONTENT_TYPE_META: Record<string, { label: string; color: string }> = {
    structured: { label: 'Table',  color: 'bg-amber-500/20 text-amber-300 border border-amber-500/30' },
    image:      { label: 'Image',  color: 'bg-sky-500/20   text-sky-300   border border-sky-500/30'   },
    text:       { label: 'Text',   color: 'bg-violet-500/20 text-violet-300 border border-violet-500/30' },
};

const LABEL_COLORS: Record<string, string> = {
    person:     'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30',
    car:        'bg-blue-500/20    text-blue-300    border border-blue-500/30',
    truck:      'bg-orange-500/20  text-orange-300  border border-orange-500/30',
    bicycle:    'bg-yellow-500/20  text-yellow-300  border border-yellow-500/30',
    motorcycle: 'bg-pink-500/20    text-pink-300    border border-pink-500/30',
};
const LABEL_COLOR_DEFAULT = 'bg-slate-500/20 text-slate-300 border border-slate-500/30';

function labelColor(label: string | undefined): string {
    return LABEL_COLORS[(label ?? '').toLowerCase()] ?? LABEL_COLOR_DEFAULT;
}

interface BBoxFrame { t: number; x: number; y: number; w: number; h: number; }
interface CvTrack { track_id: string; global_track_id?: string; label?: string; bboxes: BBoxFrame[]; }

/** Compact table showing detected tracks from cv_meta. */
function CvMetaTracks({ raw }: { raw: string }) {
    let tracks: CvTrack[] = [];
    try { tracks = JSON.parse(raw); } catch { return null; }
    if (!Array.isArray(tracks) || tracks.length === 0) return null;

    return (
        <div className="border-t border-slate-800/50 pt-3 mt-1">
            <p className="text-xs text-slate-500 mb-2">Detected tracks</p>
            <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse">
                    <thead>
                        <tr className="text-slate-500">
                            <th className="text-left px-2 py-1 font-medium">Global ID</th>
                            <th className="text-left px-2 py-1 font-medium">Label</th>
                            <th className="text-left px-2 py-1 font-medium">Seg ID</th>
                            <th className="text-right px-2 py-1 font-medium">First seen</th>
                            <th className="text-right px-2 py-1 font-medium">Last seen</th>
                            <th className="text-right px-2 py-1 font-medium">Frames</th>
                        </tr>
                    </thead>
                    <tbody>
                        {tracks.map((tr, i) => {
                            const times = tr.bboxes.map(b => b.t).sort((a, b) => a - b);
                            const first = times[0] ?? 0;
                            const last = times[times.length - 1] ?? 0;
                            return (
                                <tr key={i} className="even:bg-slate-800/20">
                                    <td className="px-2 py-1">
                                        <span className="font-mono text-violet-400">{tr.global_track_id ?? '—'}</span>
                                    </td>
                                    <td className="px-2 py-1">
                                        <span className={`px-1.5 py-0.5 rounded text-xs ${labelColor(tr.label)}`}>
                                            {tr.label ?? 'unknown'}
                                        </span>
                                    </td>
                                    <td className="px-2 py-1 font-mono text-slate-500">{tr.track_id}</td>
                                    <td className="px-2 py-1 text-right text-slate-400">{first.toFixed(2)}s</td>
                                    <td className="px-2 py-1 text-right text-slate-400">{last.toFixed(2)}s</td>
                                    <td className="px-2 py-1 text-right text-slate-400">{tr.bboxes.length}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

/** Lazy video player: fetches presigned URL on demand, renders <video> once available. */
function VideoClipPlayer({ collection, pointId }: { collection: string; pointId: string }) {
    const [url, setUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await clipsApi.getUrl(collection, pointId);
            setUrl(res.data.url);
        } catch {
            setError('Could not load clip.');
        } finally {
            setLoading(false);
        }
    }, [collection, pointId]);

    if (error) return <p className="text-xs text-red-400">{error}</p>;

    if (!url) {
        return (
            <button
                type="button"
                onClick={load}
                disabled={loading}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-violet-500/20 text-violet-300 hover:bg-violet-500/30 text-xs disabled:opacity-50"
            >
                {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                {loading ? 'Loading clip…' : 'Play clip'}
            </button>
        );
    }

    return (
        <video
            src={url}
            controls
            className="rounded-lg max-w-full border border-slate-700/40"
            style={{ maxHeight: 280 }}
        />
    );
}

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
                            const contentType = item.payload.content_type as string | undefined;
                            const tableContent = item.payload.table_content as string | undefined;
                            const imageData = item.payload.image_data as string | undefined;
                            const imageCaption = item.payload.image_caption as string | undefined;
                            const tStart = item.payload.t_start as number | undefined;
                            const tEnd = item.payload.t_end as number | undefined;
                            const pageNo = item.payload.page_number ?? item.payload.page;
                            const textType = item.payload.text_type as string | undefined;
                            const cvMeta = item.payload.cv_meta as string | undefined;
                            const hasClip = typeof item.payload.clip_key === 'string';
                            const timeRange =
                                typeof tStart === 'number' && typeof tEnd === 'number'
                                    ? `${formatTime(tStart)} – ${formatTime(tEnd)}`
                                    : null;
                            const preview = text.length <= PREVIEW_LENGTH ? text : text.slice(0, PREVIEW_LENGTH) + '…';
                            const metadataEntries = getMetadataEntries(item.payload);
                            const ctMeta = contentType ? (CONTENT_TYPE_META[contentType] ?? null) : null;
                            const ChunkIcon = isVideo ? Video : contentType === 'structured' ? Table2 : contentType === 'image' ? Image : FileText;

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
                                            {/* Header row */}
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-800/50 flex items-center justify-center">
                                                    <ChunkIcon className="w-4 h-4 text-violet-400" />
                                                </div>
                                                <code className="text-xs text-slate-400 font-mono truncate max-w-[160px]" title={item.id}>
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
                                                {ctMeta && (
                                                    <span className={`text-xs px-2 py-0.5 rounded-lg font-medium ${ctMeta.color}`}>
                                                        {ctMeta.label}
                                                    </span>
                                                )}
                                                {textType && (
                                                    <span className="text-xs text-slate-500 capitalize">{textType}</span>
                                                )}
                                                {pageNo != null && (
                                                    <span className="text-xs text-slate-500">p.{String(pageNo)}</span>
                                                )}
                                                {timeRange && (
                                                    <span className="text-xs text-violet-400">{timeRange}</span>
                                                )}
                                            </div>

                                            {/* Video clip player */}
                                            {isVideo && hasClip && (
                                                <VideoClipPlayer
                                                    collection={domain.qdrant_collection}
                                                    pointId={item.id}
                                                />
                                            )}

                                            {/* Content area — type-aware rendering */}
                                            {contentType === 'image' && imageData ? (
                                                <div className="space-y-2">
                                                    <img
                                                        src={`data:image/jpeg;base64,${imageData}`}
                                                        alt={imageCaption ?? 'Extracted image'}
                                                        className="rounded-lg max-w-full max-h-64 object-contain border border-slate-700/40"
                                                    />
                                                    {imageCaption && (
                                                        <p className="text-xs text-slate-400 italic">{imageCaption}</p>
                                                    )}
                                                </div>
                                            ) : contentType === 'structured' && tableContent ? (
                                                <div className="overflow-x-auto">
                                                    <MarkdownTable md={tableContent} />
                                                </div>
                                            ) : (
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
                                            )}

                                            {/* Detected tracks from cv_meta */}
                                            {cvMeta && typeof cvMeta === 'string' && (
                                                <CvMetaTracks raw={cvMeta} />
                                            )}

                                            {/* Metadata pills */}
                                            {metadataEntries.length > 0 && (
                                                <div className="border-t border-slate-800/50 pt-3 mt-1">
                                                    <p className="text-xs text-slate-500 mb-2">Metadata</p>
                                                    <div className="flex flex-wrap gap-2">
                                                        {metadataEntries.map(([key, value]) => {
                                                            const display =
                                                                Array.isArray(value)
                                                                    ? `[${(value as unknown[]).map(v => typeof v === 'number' ? v.toFixed(1) : String(v)).join(', ')}]`
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

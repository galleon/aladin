import { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
    videoTranscriptionApi,
    agentsApi,
    Agent,
    TranscriptionJobListItem,
    TranscriptionJobDetail,
} from '../api/client';
import { Upload, Loader2, FileText, Download, AlertCircle, CheckCircle2, Clock, XCircle, List } from 'lucide-react';

interface VideoTranscriptionProps {
    agentId: number;
}

const POLL_INTERVAL_MS = 4000;
const STATUS_LABELS: Record<string, string> = {
    pending: 'Queued',
    processing: 'Processing...',
    completed: 'Completed',
    failed: 'Failed',
};

export default function VideoTranscription({ agentId }: VideoTranscriptionProps) {
    const [searchParams, setSearchParams] = useSearchParams();
    const jobIdFromUrl = searchParams.get('job');
    const selectedJobId = jobIdFromUrl ? parseInt(jobIdFromUrl, 10) : null;

    const [file, setFile] = useState<File | null>(null);
    const [language, setLanguage] = useState<string>('');
    const [subtitleLanguage, setSubtitleLanguage] = useState<string>('');
    const [translationAgents, setTranslationAgents] = useState<Agent[]>([]);
    const [selectedTranslationAgentId, setSelectedTranslationAgentId] = useState<number | null>(null);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [isAddingSubtitles, setIsAddingSubtitles] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [jobQueuedMessage, setJobQueuedMessage] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [jobs, setJobs] = useState<TranscriptionJobListItem[]>([]);
    const [jobsLoading, setJobsLoading] = useState(true);
    const [selectedJobDetail, setSelectedJobDetail] = useState<TranscriptionJobDetail | null>(null);
    const [detailLoading, setDetailLoading] = useState(false);

    // Sync URL ?job= with selected job
    const setSelectedJob = (jobId: number | null) => {
        setSearchParams(
            (prev) => {
                const next = new URLSearchParams(prev);
                if (jobId == null) next.delete('job');
                else next.set('job', String(jobId));
                return next;
            },
            { replace: true }
        );
    };

    // Fetch jobs list (all transcription jobs for the logged user)
    const fetchJobs = async () => {
        try {
            const res = await videoTranscriptionApi.listJobs();
            setJobs(Array.isArray(res.data) ? res.data : []);
        } catch (err) {
            console.error('Failed to fetch transcription jobs:', err);
            setJobs([]);
        } finally {
            setJobsLoading(false);
        }
    };

    useEffect(() => {
        setJobsLoading(true);
        fetchJobs();
    }, []);

    // When URL has ?job=, ensure we have that job selected and load detail
    useEffect(() => {
        if (selectedJobId == null) {
            setSelectedJobDetail(null);
            return;
        }
        let cancelled = false;
        setDetailLoading(true);
        videoTranscriptionApi
            .getJob(selectedJobId)
            .then((res) => {
                if (!cancelled) setSelectedJobDetail(res.data);
            })
            .catch(() => {
                if (!cancelled) setSelectedJobDetail(null);
            })
            .finally(() => {
                if (!cancelled) setDetailLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [selectedJobId]);

    // Poll selected job while it's pending or processing
    useEffect(() => {
        if (!selectedJobDetail || !['pending', 'processing'].includes(selectedJobDetail.status)) return;
        const t = setInterval(async () => {
            try {
                const res = await videoTranscriptionApi.getJob(selectedJobDetail.id);
                setSelectedJobDetail(res.data);
                if (!['pending', 'processing'].includes(res.data.status)) {
                    fetchJobs();
                }
            } catch {
                // ignore
            }
        }, POLL_INTERVAL_MS);
        return () => clearInterval(t);
    }, [selectedJobDetail?.id, selectedJobDetail?.status]);

    // Fetch translation agents on mount
    useEffect(() => {
        const fetchTranslationAgents = async () => {
            try {
                const response = await agentsApi.list('translation');
                setTranslationAgents(response.data);
                if (response.data.length > 0) {
                    setSelectedTranslationAgentId(response.data[0].id);
                }
            } catch (err) {
                console.error('Failed to fetch translation agents:', err);
            }
        };
        fetchTranslationAgents();
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            const validTypes = [
                'video/mp4',
                'video/mpeg',
                'video/quicktime',
                'video/x-msvideo',
                'audio/mpeg',
                'audio/mp3',
                'audio/wav',
                'audio/webm',
            ];
            if (!validTypes.includes(selectedFile.type)) {
                setError('Please select a valid video or audio file (MP4, MP3, WAV, etc.)');
                return;
            }
            setFile(selectedFile);
            setError(null);
            setJobQueuedMessage(null);
        }
    };

    const handleTranscribe = async () => {
        if (!file) {
            setError('Please select a file first');
            return;
        }
        setIsTranscribing(true);
        setError(null);
        setJobQueuedMessage(null);
        try {
            const response = await videoTranscriptionApi.transcribe(
                agentId,
                file,
                language || undefined
            );
            const { job_id } = response.data;
            setJobQueuedMessage(`Job queued. You can track it in the list on the left.`);
            setSelectedJob(job_id);
            await fetchJobs();
            const detailRes = await videoTranscriptionApi.getJob(job_id);
            setSelectedJobDetail(detailRes.data);
        } catch (err: unknown) {
            const ax = err as { response?: { data?: { detail?: string } }; message?: string };
            setError(ax.response?.data?.detail || ax.message || 'Transcription request failed');
        } finally {
            setIsTranscribing(false);
        }
    };

    const handleAddSubtitles = async () => {
        if (!file) {
            setError('Please select a file first');
            return;
        }
        setIsAddingSubtitles(true);
        setError(null);
        setJobQueuedMessage(null);
        try {
            const response = await videoTranscriptionApi.addSubtitles(
                agentId,
                file,
                language || undefined,
                subtitleLanguage || undefined,
                selectedTranslationAgentId || undefined
            );
            const { job_id } = response.data;
            setJobQueuedMessage(`Job queued. You can track it in the list on the left.`);
            setSelectedJob(job_id);
            await fetchJobs();
            const detailRes = await videoTranscriptionApi.getJob(job_id);
            setSelectedJobDetail(detailRes.data);
        } catch (err: unknown) {
            const ax = err as { response?: { data?: { detail?: string } }; message?: string };
            setError(ax.response?.data?.detail || ax.message || 'Add subtitles request failed');
        } finally {
            setIsAddingSubtitles(false);
        }
    };

    const handleDownload = async () => {
        if (!selectedJobDetail || selectedJobDetail.job_type !== 'add_subtitles' || selectedJobDetail.status !== 'completed') return;
        try {
            const res = await videoTranscriptionApi.downloadVideo(selectedJobDetail.id);
            const url = window.URL.createObjectURL(res.data as Blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = (selectedJobDetail.source_filename || 'video').replace(/\.[^/.]+$/, '') + '_with_subtitles.mp4';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (err) {
            setError('Download failed');
        }
    };

    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const formatDate = (iso: string) => {
        try {
            const d = new Date(iso);
            return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        } catch {
            return iso;
        }
    };

    const transcription = selectedJobDetail?.result_transcript ?? null;

    return (
        <div className="flex gap-6">
            {/* Left sidebar: past requests */}
            <div className="w-64 shrink-0 flex flex-col">
                <div className="flex items-center gap-2 mb-3">
                    <List className="w-5 h-5 text-slate-400" />
                    <h2 className="text-sm font-semibold text-slate-300">Past requests</h2>
                </div>
                <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-xl overflow-hidden flex-1 min-h-0 flex flex-col">
                    {jobsLoading ? (
                        <div className="p-4 flex items-center justify-center text-slate-500">
                            <Loader2 className="w-5 h-5 animate-spin" />
                        </div>
                    ) : jobs.length === 0 ? (
                        <p className="p-4 text-sm text-slate-500">No jobs yet</p>
                    ) : (
                        <ul className="overflow-y-auto flex-1 p-2 space-y-1">
                            {jobs.map((job) => {
                                const isSelected = selectedJobId === job.id;
                                const statusIcon =
                                    job.status === 'completed' ? (
                                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                                    ) : job.status === 'failed' ? (
                                        <XCircle className="w-4 h-4 text-red-400 shrink-0" />
                                    ) : (
                                        <Clock className="w-4 h-4 text-slate-400 shrink-0" />
                                    );
                                return (
                                    <li key={job.id}>
                                        <button
                                            type="button"
                                            onClick={() => setSelectedJob(job.id)}
                                            className={`w-full text-left px-3 py-2 rounded-lg flex items-start gap-2 transition-colors ${isSelected
                                                    ? 'bg-violet-600/30 text-white border border-violet-500/50'
                                                    : 'hover:bg-slate-800/50 text-slate-300'
                                                }`}
                                        >
                                            {statusIcon}
                                            <div className="min-w-0 flex-1">
                                                <p className="text-xs font-medium truncate" title={job.source_filename}>
                                                    {job.source_filename}
                                                </p>
                                                <p className="text-xs text-slate-500">
                                                    {STATUS_LABELS[job.status] ?? job.status} · {formatDate(job.created_at)}
                                                </p>
                                            </div>
                                        </button>
                                    </li>
                                );
                            })}
                        </ul>
                    )}
                </div>
            </div>

            {/* Main: upload form + selected job detail */}
            <div className="flex-1 min-w-0 space-y-6">
                {jobQueuedMessage && (
                    <div className="flex items-center gap-2 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-emerald-400 text-sm">
                        <CheckCircle2 className="w-5 h-5 shrink-0" />
                        {jobQueuedMessage}
                    </div>
                )}

                {/* File Upload */}
                <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
                    <h2 className="text-lg font-semibold text-white mb-4">Upload Video/Audio</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">Select File</label>
                            <div className="flex items-center gap-4">
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="video/*,audio/*"
                                    onChange={handleFileSelect}
                                    className="hidden"
                                />
                                <button
                                    onClick={() => fileInputRef.current?.click()}
                                    className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors"
                                >
                                    <Upload className="w-5 h-5" />
                                    Choose File
                                </button>
                                {file && (
                                    <div className="flex items-center gap-2 text-slate-300">
                                        <FileText className="w-5 h-5" />
                                        <span className="text-sm">{file.name}</span>
                                        <span className="text-xs text-slate-500">
                                            ({(file.size / 1024 / 1024).toFixed(2)} MB)
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                    Source Language (optional)
                                </label>
                                <input
                                    type="text"
                                    value={language}
                                    onChange={(e) => setLanguage(e.target.value)}
                                    placeholder="e.g., en, fr, es"
                                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                    Subtitle Language (optional)
                                </label>
                                <input
                                    type="text"
                                    value={subtitleLanguage}
                                    onChange={(e) => setSubtitleLanguage(e.target.value)}
                                    placeholder="e.g., es, de, fr"
                                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500"
                                />
                            </div>
                        </div>
                        {subtitleLanguage && (
                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">
                                    Translation Agent
                                </label>
                                <select
                                    value={selectedTranslationAgentId ?? ''}
                                    onChange={(e) =>
                                        setSelectedTranslationAgentId(
                                            e.target.value ? parseInt(e.target.value, 10) : null
                                        )
                                    }
                                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                                >
                                    <option value="">-- Select Translation Agent --</option>
                                    {translationAgents.map((agent) => (
                                        <option key={agent.id} value={agent.id}>
                                            {agent.name} {agent.llm_model ? `(${agent.llm_model})` : ''}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        )}
                        {error && (
                            <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400">
                                <AlertCircle className="w-5 h-5" />
                                <span className="text-sm">{error}</span>
                            </div>
                        )}
                        <div className="flex gap-3">
                            <button
                                onClick={handleTranscribe}
                                disabled={!file || isTranscribing || isAddingSubtitles}
                                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isTranscribing ? (
                                    <>
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                        Queuing...
                                    </>
                                ) : (
                                    <>
                                        <FileText className="w-5 h-5" />
                                        Transcribe Only
                                    </>
                                )}
                            </button>
                            <button
                                onClick={handleAddSubtitles}
                                disabled={!file || isTranscribing || isAddingSubtitles}
                                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isAddingSubtitles ? (
                                    <>
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                        Queuing...
                                    </>
                                ) : (
                                    <>
                                        <Download className="w-5 h-5" />
                                        Transcribe & Add Subtitles
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Selected job detail */}
                {selectedJobId != null && (
                    <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-white mb-4">Job detail</h2>
                        {detailLoading ? (
                            <div className="flex items-center gap-2 text-slate-400">
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Loading...
                            </div>
                        ) : selectedJobDetail ? (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between flex-wrap gap-2">
                                    <div className="flex items-center gap-2">
                                        <span
                                            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${selectedJobDetail.status === 'completed'
                                                    ? 'bg-emerald-500/20 text-emerald-400'
                                                    : selectedJobDetail.status === 'failed'
                                                        ? 'bg-red-500/20 text-red-400'
                                                        : 'bg-slate-500/20 text-slate-400'
                                                }`}
                                        >
                                            {selectedJobDetail.status === 'processing' && (
                                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                            )}
                                            {STATUS_LABELS[selectedJobDetail.status] ?? selectedJobDetail.status}
                                        </span>
                                        <span className="text-sm text-slate-400">
                                            {selectedJobDetail.source_filename}
                                        </span>
                                    </div>
                                    {selectedJobDetail.job_type === 'add_subtitles' &&
                                        selectedJobDetail.status === 'completed' && (
                                            <button
                                                onClick={handleDownload}
                                                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white font-medium rounded-lg"
                                            >
                                                <Download className="w-4 h-4" />
                                                Download Video
                                            </button>
                                        )}
                                </div>
                                {selectedJobDetail.error_message && (
                                    <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                                        {selectedJobDetail.error_message}
                                    </div>
                                )}
                                {!transcription && ['pending', 'processing'].includes(selectedJobDetail.status) && (
                                    <p className="text-slate-500 text-sm">
                                        Job in progress. The transcript will appear here when the job completes.
                                    </p>
                                )}
                                {transcription && (
                                    <>
                                        <div className="flex items-center gap-2 text-sm text-slate-400">
                                            {transcription.language && (
                                                <span>
                                                    Language: {transcription.language}
                                                    {transcription.language_probability != null &&
                                                        ` (${Math.round(transcription.language_probability * 100)}%)`}
                                                </span>
                                            )}
                                        </div>
                                        <div className="mb-4">
                                            <h3 className="text-sm font-medium text-slate-400 mb-2">Full Transcript</h3>
                                            <div className="bg-slate-800/50 rounded-xl p-4 max-h-96 overflow-y-auto">
                                                <p className="text-slate-200 whitespace-pre-wrap leading-relaxed">
                                                    {transcription.transcript}
                                                </p>
                                            </div>
                                        </div>
                                        {transcription.segments && transcription.segments.length > 0 && (
                                            <div>
                                                <h3 className="text-sm font-medium text-slate-400 mb-2">Segments</h3>
                                                <div className="space-y-2 max-h-96 overflow-y-auto">
                                                    {transcription.segments.map((segment, idx) => (
                                                        <div
                                                            key={idx}
                                                            className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/50"
                                                        >
                                                            <div className="flex items-center gap-2 mb-1">
                                                                <span className="text-xs text-slate-500 font-mono">
                                                                    {formatTime(segment.start)} →{' '}
                                                                    {formatTime(segment.end)}
                                                                </span>
                                                            </div>
                                                            <p className="text-slate-200 text-sm">{segment.text}</p>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        ) : (
                            <p className="text-slate-500">Job not found.</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

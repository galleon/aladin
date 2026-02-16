import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { jobsApi, UnifiedJobListItem, QueueInfo } from '../api/client';
import { Loader2, Zap, AlertCircle, RotateCcw, XCircle, Trash2, RefreshCw } from 'lucide-react';

const STATUS_LABELS: Record<string, string> = {
    pending: 'Queued',
    queued: 'Queued',
    processing: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled',
    crawling: 'Crawling',
    extracting: 'Extracting',
    partitioning: 'Partitioning',
    embedding: 'Embedding',
    translating: 'Translating',
    generating: 'Generating',
};

const JOB_TYPE_LABELS: Record<string, string> = {
    transcription: 'Transcription',
    translation: 'Translation',
    ingestion_web: 'Web ingestion',
    ingestion_file: 'File ingestion',
    ingestion_video: 'Video ingestion',
};

function formatDate(iso: string | null | undefined): string {
    if (!iso) return '—';
    try {
        return new Date(iso).toLocaleString(undefined, {
            dateStyle: 'short',
            timeStyle: 'short',
        });
    } catch {
        return iso;
    }
}

function formatDuration(seconds: number | null | undefined): string {
    if (seconds == null || seconds < 0) return '—';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
    const h = Math.floor(m / 60);
    const min = m % 60;
    return min > 0 ? `${h}h ${min}m` : `${h}h`;
}

export default function JobsPage() {
    const queryClient = useQueryClient();

    const { data: jobs = [], isLoading, error, refetch: refetchJobs } = useQuery({
        queryKey: ['unifiedJobs'],
        queryFn: () => jobsApi.listJobs({ limit: 200 }).then((res) => res.data),
        refetchInterval: (query) => {
            const data = query.state.data as UnifiedJobListItem[] | undefined;
            const hasActive = data?.some(
                (j) => j.status === 'queued' || j.status === 'pending' || j.status === 'processing' || j.status === 'extracting' || j.status === 'embedding'
            );
            return hasActive ? 5000 : 15000;
        },
    });

    const { data: queuesData, refetch: refetchQueues } = useQuery({
        queryKey: ['jobsQueues'],
        queryFn: () => jobsApi.listQueues().then((res) => res.data),
        refetchInterval: (query) => {
            const data = query.state.data as { queues?: { count?: number }[] } | undefined;
            const pending = data?.queues?.[0]?.count ?? 0;
            return pending > 0 ? 5000 : 15000;
        },
        retry: false,
    });

    const { data: queueStatus, refetch: refetchQueueStatus } = useQuery({
        queryKey: ['jobsQueueStatus'],
        queryFn: () => jobsApi.getQueueStatus().then((res) => res.data),
        refetchInterval: (query) => {
            const data = query.state.data as { queued_count?: number } | undefined;
            const pending = data?.queued_count ?? 0;
            return pending > 0 ? 5000 : 15000;
        },
        retry: false,
    });

    const handleRefresh = () => {
        refetchJobs();
        refetchQueues();
        refetchQueueStatus();
    };

    const purgeQueueMutation = useMutation({
        mutationFn: (queueName: string) => jobsApi.purgeQueue(queueName),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobsQueues'] });
            queryClient.invalidateQueries({ queryKey: ['jobsQueueStatus'] });
            queryClient.invalidateQueries({ queryKey: ['unifiedJobs'] });
        },
    });

    const requeueMutation = useMutation({
        mutationFn: (jobId: number) => jobsApi.requeue(jobId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['unifiedJobs'] });
            queryClient.invalidateQueries({ queryKey: ['jobsQueues'] });
            queryClient.invalidateQueries({ queryKey: ['jobsQueueStatus'] });
        },
    });

    const cancelMutation = useMutation({
        mutationFn: (jobId: number) => jobsApi.cancel(jobId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['unifiedJobs'] });
        },
    });

    const deleteMutation = useMutation({
        mutationFn: (jobId: number) => jobsApi.delete(jobId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['unifiedJobs'] });
            queryClient.invalidateQueries({ queryKey: ['dataDomain'] });
        },
    });

    const queues: QueueInfo[] = queuesData?.queues ?? [];
    const queuedCount = queueStatus?.queued_count ?? queues[0]?.count ?? null;
    const queueName = queueStatus?.queue_name ?? queues[0]?.name ?? 'arq:queue';

    if (isLoading) {
        return (
            <div className="flex justify-center py-12">
                <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="max-w-4xl mx-auto">
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-2 text-red-400">
                    <AlertCircle className="w-5 h-5 shrink-0" />
                    <span>Failed to load jobs</span>
                </div>
            </div>
        );
    }

    return (
        <div className="max-w-6xl mx-auto space-y-6">
            <div>
                <h1 className="text-3xl font-bold text-white mb-2">Jobs</h1>
                <p className="text-slate-400">
                    Queue status and unified view of all jobs (transcription, translation, ingestion).
                </p>
            </div>

            {/* Queue status */}
            <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-4">
                <div className="flex items-center justify-between mb-3">
                    <h2 className="text-sm font-semibold text-slate-300">Queue (Redis)</h2>
                    <button
                        type="button"
                        onClick={handleRefresh}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded-lg transition-colors"
                        title="Refresh queue and jobs"
                    >
                        <RefreshCw className="w-4 h-4" />
                        Refresh
                    </button>
                </div>
                <div className="flex flex-wrap items-center gap-4">
                    <span className="text-slate-300 font-medium">{queues[0]?.label ?? 'Job queue'}</span>
                    <code className="text-xs text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
                        {queueName}
                    </code>
                    <span className="text-slate-400 text-sm">
                        {queuedCount != null
                            ? `${queuedCount} pending`
                            : queueStatus?.error
                                ? `— (${queueStatus.error})`
                                : '—'}
                    </span>
                    {(queuedCount === 0 && jobs.some((j: UnifiedJobListItem) => j.status === 'queued' || j.status === 'pending')) && (
                        <span className="text-amber-400/90 text-sm">
                            Queue empty but some jobs show as Queued—use Re-queue to send them to a worker.
                        </span>
                    )}
                    {queues.length > 0 && (
                        <button
                            type="button"
                            onClick={() => {
                                if (confirm('Purge the job queue? This removes all pending jobs.')) {
                                    purgeQueueMutation.mutate(queues[0].name);
                                }
                            }}
                            disabled={purgeQueueMutation.isPending || (queuedCount ?? 0) === 0}
                            className="flex items-center gap-1.5 px-2.5 py-1.5 text-sm bg-amber-600/20 text-amber-400 hover:bg-amber-600/30 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Remove all pending jobs from the queue"
                        >
                            {purgeQueueMutation.isPending ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <Zap className="w-4 h-4" />
                            )}
                            Purge queue
                        </button>
                    )}
                </div>
            </div>

            {/* Unified jobs table */}
            <div>
                <h2 className="text-sm font-semibold text-slate-300 mb-3">Jobs</h2>
                <p className="text-slate-500 text-xs mb-3">
                    All jobs from database
                </p>

                {jobs.length === 0 ? (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-12 text-center">
                        <p className="text-slate-400">No jobs yet.</p>
                    </div>
                ) : (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead>
                                    <tr className="border-b border-slate-800/50 bg-slate-800/30">
                                        <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                            Job ID
                                        </th>
                                        <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                            Job type
                                        </th>
                                        <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                            Status
                                        </th>
                                        <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                            Date submitted
                                        </th>
                                        <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                            Duration
                                        </th>
                                        <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                            Actions
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {jobs.map((job: UnifiedJobListItem) => {
                                        const canRequeue =
                                            (job.status === 'pending' || job.status === 'queued' || job.status === 'failed') &&
                                            (job.job_type === 'ingestion_file' || job.job_type === 'ingestion_video');
                                        const canCancel =
                                            job.status !== 'completed' &&
                                            job.status !== 'cancelled';
                                        return (
                                            <tr
                                                key={job.id}
                                                className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
                                            >
                                                <td className="px-4 py-3 text-slate-400 font-mono text-sm">
                                                    {job.id}
                                                </td>
                                                <td className="px-4 py-3 text-slate-300">
                                                    {JOB_TYPE_LABELS[job.job_type] ?? job.job_type}
                                                </td>
                                                <td className="px-4 py-3 text-slate-300">
                                                    {STATUS_LABELS[job.status] ?? job.status}
                                                </td>
                                                <td className="px-4 py-3 text-slate-400 text-sm">
                                                    {formatDate(job.created_at)}
                                                </td>
                                                <td className="px-4 py-3 text-slate-400 text-sm">
                                                    {formatDuration(job.duration_seconds)}
                                                </td>
                                                <td className="px-4 py-3">
                                                    <div className="flex items-center gap-1">
                                                        {canRequeue && (
                                                            <button
                                                                type="button"
                                                                onClick={() => {
                                                                    if (confirm('Re-queue this job so a worker picks it up?')) {
                                                                        requeueMutation.mutate(job.id);
                                                                    }
                                                                }}
                                                                disabled={requeueMutation.isPending}
                                                                className="p-1.5 text-slate-400 hover:text-amber-400 hover:bg-amber-500/10 rounded-lg disabled:opacity-50"
                                                                title="Re-queue job"
                                                            >
                                                                <RotateCcw className="w-4 h-4" />
                                                            </button>
                                                        )}
                                                        {canCancel && (
                                                            <button
                                                                type="button"
                                                                onClick={() => {
                                                                    if (confirm('Cancel this job?')) {
                                                                        cancelMutation.mutate(job.id);
                                                                    }
                                                                }}
                                                                disabled={cancelMutation.isPending}
                                                                className="p-1.5 text-slate-400 hover:text-amber-400 hover:bg-amber-500/10 rounded-lg disabled:opacity-50"
                                                                title="Cancel job"
                                                            >
                                                                <XCircle className="w-4 h-4" />
                                                            </button>
                                                        )}
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                if (confirm('Delete this job record from the database?')) {
                                                                    deleteMutation.mutate(job.id);
                                                                }
                                                            }}
                                                            disabled={deleteMutation.isPending}
                                                            className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg disabled:opacity-50"
                                                            title="Delete job record"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

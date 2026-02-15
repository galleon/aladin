import { useState, useRef, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { agentsApi, translationApi } from '../api/client';
import {
  Languages,
  Loader2,
  ArrowLeft,
  Upload,
  FileText,
  Download,
  CheckCircle,
  XCircle,
  AlertCircle,
  ToggleLeft,
  ToggleRight,
  RefreshCw,
  ArrowRight,
  Copy,
  Check,
  Trash2,
  Folder,
  Clock,
} from 'lucide-react';

export default function TranslationChat() {
  const { agentId } = useParams<{ agentId: string }>();
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [simplified, setSimplified] = useState(false);
  const [activeTab, setActiveTab] = useState<'text' | 'file'>('text');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [copied, setCopied] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch agent details
  const { data: agent, isLoading: agentLoading } = useQuery({
    queryKey: ['agent', agentId],
    queryFn: () => agentsApi.get(parseInt(agentId!)).then(res => res.data),
    enabled: !!agentId,
  });

  // Fetch supported languages
  const { data: languagesData } = useQuery({
    queryKey: ['supportedLanguages'],
    queryFn: () => translationApi.getSupportedLanguages().then(res => res.data),
  });

  // Fetch translation jobs with smart polling
  const { data: jobs, refetch: refetchJobs } = useQuery({
    queryKey: ['translationJobs', agentId],
    queryFn: () => translationApi.listJobs(parseInt(agentId!)).then(res => res.data),
    enabled: !!agentId,
    // Poll faster (2s) when jobs are active, slower (10s) when idle
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data || data.length === 0) return 10000;
      const hasActiveJobs = data.some((job: { status: string }) =>
        ['pending', 'extracting', 'translating', 'generating'].includes(job.status)
      );
      return hasActiveJobs ? 2000 : 10000;
    },
    refetchOnWindowFocus: true,
  });

  // Set default target language from agent
  useEffect(() => {
    if (agent?.target_language) {
      setTargetLanguage(agent.target_language);
    }
  }, [agent]);

  // Translation mutation (direct, not chat)
  const translateMutation = useMutation({
    mutationFn: (text: string) =>
      translationApi.translateText(parseInt(agentId!), {
        text,
        target_language: targetLanguage,
        simplified,
        source_language: 'auto',
      }),
    onSuccess: (response) => {
      setTranslatedText(response.data.translated_text);
    },
  });

  // File upload mutation
  const uploadMutation = useMutation({
    mutationFn: (file: File) =>
      translationApi.translateFile(parseInt(agentId!), file, targetLanguage, simplified),
    onSuccess: () => {
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      refetchJobs();
    },
  });

  const handleTranslate = () => {
    if (!sourceText.trim() || translateMutation.isPending) return;
    translateMutation.mutate(sourceText);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleFileUpload = () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
  };

  const handleDownload = async (jobId: number, filename: string) => {
    try {
      const response = await translationApi.downloadTranslation(jobId);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const handleCopy = async () => {
    if (translatedText) {
      await navigator.clipboard.writeText(translatedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleClear = () => {
    setSourceText('');
    setTranslatedText('');
  };

  const getJobStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'pending':
      case 'extracting':
      case 'translating':
      case 'generating':
        return <Loader2 className="w-5 h-5 text-violet-400 animate-spin" />;
      default:
        return <AlertCircle className="w-5 h-5 text-amber-400" />;
    }
  };

  const getJobStatusLabel = (status: string, progress: number) => {
    switch (status) {
      case 'pending':
        return 'Waiting to start...';
      case 'extracting':
        return `Extracting text from document... (${progress}%)`;
      case 'translating':
        return `Translating content... (${progress}%)`;
      case 'generating':
        return `Generating PDF... (${progress}%)`;
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      default:
        return `${status} (${progress}%)`;
    }
  };

  // Format processing time in human-readable format
  const formatProcessingTime = (seconds: number | null): string => {
    if (seconds === null || seconds === undefined) return '';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  };

  // Extract short directory name from path
  const getShortDirectory = (path: string | null): string => {
    if (!path) return '';
    const parts = path.split('/');
    // Get last 2 parts (user_id/job_uuid)
    return parts.slice(-2).join('/');
  };

  if (agentLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-200px)]">
        <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
      </div>
    );
  }

  if (!agent || agent.agent_type !== 'translation') {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-200px)]">
        <Languages className="w-16 h-16 text-slate-600 mb-4" />
        <h2 className="text-xl font-semibold text-white mb-2">Agent not found</h2>
        <p className="text-slate-400 mb-4">This translation agent doesn't exist.</p>
        <Link to="/agents" className="text-violet-400 hover:text-violet-300">
          ← Back to agents
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto h-[calc(100vh-160px)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Link
            to="/agents"
            className="p-2 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20">
                <Languages className="w-5 h-5 text-emerald-400" />
              </div>
              <h1 className="text-2xl font-bold text-white">{agent.name}</h1>
            </div>
            <p className="text-slate-400 text-sm mt-1">{agent.description || 'Translation Agent'}</p>
          </div>
        </div>

        {/* Translation options */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-400">Target:</span>
            <select
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
            >
              {languagesData && Object.entries(languagesData.languages).map(([code, name]) => (
                <option key={code} value={code}>{name}</option>
              ))}
            </select>
          </div>
          <button
            onClick={() => setSimplified(!simplified)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              simplified
                ? 'bg-emerald-500/20 text-emerald-400'
                : 'bg-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            {simplified ? <ToggleRight className="w-5 h-5" /> : <ToggleLeft className="w-5 h-5" />}
            Simplified
          </button>
        </div>
      </div>

      {/* Tab selector */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setActiveTab('text')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'text'
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:text-white'
          }`}
        >
          <Languages className="w-4 h-4" />
          Text Translation
        </button>
        <button
          onClick={() => setActiveTab('file')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'file'
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-800 text-slate-400 hover:text-white'
          }`}
        >
          <FileText className="w-4 h-4" />
          File Translation
        </button>
      </div>

      {/* Text Translation Tab - Side by Side */}
      {activeTab === 'text' && (
        <div className="flex-1 flex flex-col gap-4">
          {/* Side by side panels */}
          <div className="flex-1 grid grid-cols-2 gap-4">
            {/* Source text panel */}
            <div className="flex flex-col bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                <span className="text-sm font-medium text-slate-400">Source Text</span>
                <button
                  onClick={handleClear}
                  disabled={!sourceText}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title="Clear"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                placeholder="Enter text to translate..."
                className="flex-1 p-4 bg-transparent text-white placeholder-slate-500 focus:outline-none resize-none text-lg leading-relaxed"
              />
              <div className="px-4 py-3 border-t border-slate-800 flex items-center justify-between">
                <span className="text-xs text-slate-500">
                  {sourceText.length} characters
                </span>
              </div>
            </div>

            {/* Arrow indicator */}
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10 hidden">
              <div className="p-3 rounded-full bg-slate-800 border border-slate-700">
                <ArrowRight className="w-5 h-5 text-emerald-400" />
              </div>
            </div>

            {/* Translated text panel */}
            <div className="flex flex-col bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
                <span className="text-sm font-medium text-emerald-400">
                  {languagesData?.languages[targetLanguage] || targetLanguage}
                  {simplified && ' (Simplified)'}
                </span>
                <button
                  onClick={handleCopy}
                  disabled={!translatedText}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title="Copy to clipboard"
                >
                  {copied ? (
                    <Check className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </button>
              </div>
              <div className="flex-1 p-4 overflow-y-auto">
                {translateMutation.isPending ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <Loader2 className="w-8 h-8 text-emerald-400 animate-spin mx-auto mb-3" />
                      <p className="text-slate-400 text-sm">Translating...</p>
                    </div>
                  </div>
                ) : translatedText ? (
                  <p className="text-white text-lg leading-relaxed whitespace-pre-wrap">
                    {translatedText}
                  </p>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-slate-500 text-center">
                      Translation will appear here
                    </p>
                  </div>
                )}
              </div>
              <div className="px-4 py-3 border-t border-slate-800 flex items-center justify-between">
                <span className="text-xs text-slate-500">
                  {translatedText.length} characters
                </span>
              </div>
            </div>
          </div>

          {/* Translate button */}
          <div className="flex justify-center">
            <button
              onClick={handleTranslate}
              disabled={!sourceText.trim() || translateMutation.isPending}
              className="flex items-center gap-3 px-8 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-emerald-500/25"
            >
              {translateMutation.isPending ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Translating...
                </>
              ) : (
                <>
                  <Languages className="w-5 h-5" />
                  Translate
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* File Translation Tab */}
      {activeTab === 'file' && (
        <div className="flex-1 flex flex-col gap-4">
          {/* Upload area */}
          <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Upload Document</h3>
            <div className="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md,.docx,.doc,.pptx,.ppt"
                onChange={handleFileSelect}
                className="hidden"
              />
              {selectedFile ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-center gap-3">
                    <FileText className="w-8 h-8 text-emerald-400" />
                    <span className="text-white font-medium">{selectedFile.name}</span>
                  </div>
                  <div className="flex justify-center gap-3">
                    <button
                      onClick={() => setSelectedFile(null)}
                      className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg hover:bg-slate-700"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleFileUpload}
                      disabled={uploadMutation.isPending}
                      className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-lg disabled:opacity-50"
                    >
                      {uploadMutation.isPending ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Uploading...
                        </>
                      ) : (
                        <>
                          <Upload className="w-4 h-4" />
                          Translate
                        </>
                      )}
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <Upload className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400 mb-2">
                    Drag and drop a file, or{' '}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="text-emerald-400 hover:text-emerald-300"
                    >
                      browse
                    </button>
                  </p>
                  <p className="text-slate-500 text-sm">
                    Supported: PDF, DOCX, PPTX, TXT, Markdown
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Jobs list */}
          <div className="flex-1 bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6 overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Translation Jobs</h3>
              <button
                onClick={() => refetchJobs()}
                className="p-2 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3">
              {jobs && jobs.length === 0 && (
                <div className="text-center py-8 text-slate-500">
                  No translation jobs yet. Upload a file to get started.
                </div>
              )}

              {jobs?.map((job) => (
                <div
                  key={job.id}
                  className="bg-slate-800/50 rounded-xl p-4"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getJobStatusIcon(job.status)}
                      <div>
                        <p className="text-white font-medium">{job.source_filename}</p>
                        <p className="text-slate-400 text-sm">
                          → {job.target_language.toUpperCase()}
                          {job.simplified_mode && ' (simplified)'}
                        </p>
                      </div>
                    </div>
                    {job.status === 'completed' && job.output_filename && (
                      <button
                        onClick={() => handleDownload(job.id, job.output_filename!)}
                        className="flex items-center gap-2 px-3 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-500"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    )}
                  </div>

                  {/* Progress bar and status */}
                  {job.status !== 'completed' && job.status !== 'failed' && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                        <span>{getJobStatusLabel(job.status, job.progress)}</span>
                        <span>{job.progress}%</span>
                      </div>
                      <div className="w-full bg-slate-700 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-emerald-500 to-teal-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {job.status === 'failed' && job.error_message && (
                    <div className="mt-2 p-2 bg-red-500/10 border border-red-500/20 rounded-lg">
                      <p className="text-red-400 text-sm">{job.error_message}</p>
                    </div>
                  )}

                  {/* Job details: directory and processing time */}
                  {(job.job_directory || job.processing_time_seconds) && (
                    <div className="mt-3 pt-3 border-t border-slate-700/50 flex flex-wrap gap-4 text-xs text-slate-500">
                      {job.job_directory && (
                        <div className="flex items-center gap-1.5">
                          <Folder className="w-3.5 h-3.5" />
                          <span className="font-mono">{getShortDirectory(job.job_directory)}</span>
                        </div>
                      )}
                      {job.processing_time_seconds && (
                        <div className="flex items-center gap-1.5">
                          <Clock className="w-3.5 h-3.5" />
                          <span>{formatProcessingTime(job.processing_time_seconds)}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

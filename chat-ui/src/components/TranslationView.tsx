import { useState, useRef, useEffect } from 'react'
import {
  Languages, Loader2, Upload, FileText, Download,
  CheckCircle, XCircle, AlertCircle, ToggleLeft, ToggleRight,
  RefreshCw, ArrowRight, Copy, Check, Trash2, Folder, Clock
} from 'lucide-react'
import { useStore } from '../store'
import api, { translateText, translateFile, getTranslationJobs, downloadTranslation } from '../api'

interface TranslationJob {
  id: number
  source_filename: string
  target_language: string
  simplified_mode: boolean
  status: string
  progress: number
  error_message: string | null
  output_filename: string | null
  job_directory: string | null
  processing_time_seconds: number | null
  created_at: string
  completed_at: string | null
}

export default function TranslationView() {
  const { selectedAgent } = useStore()
  const [sourceText, setSourceText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [targetLanguage, setTargetLanguage] = useState('en')
  const [simplified, setSimplified] = useState(false)
  const [activeTab, setActiveTab] = useState<'text' | 'file'>('text')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [copied, setCopied] = useState(false)
  const [jobs, setJobs] = useState<TranslationJob[]>([])
  const [languages, setLanguages] = useState<Record<string, string>>({})
  const [isTranslating, setIsTranslating] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const jobsPollInterval = useRef<number | null>(null)

  // Fetch supported languages
  useEffect(() => {
    api.get('/translation/languages')
      .then(res => {
        setLanguages(res.data.languages || {})
      })
      .catch(() => {
        // Fallback to default languages if API fails
        setLanguages({
          en: 'English',
          fr: 'French',
          de: 'German',
          es: 'Spanish',
          it: 'Italian',
          pt: 'Portuguese',
          zh: 'Chinese',
          ja: 'Japanese',
          ko: 'Korean',
        })
      })
  }, [])

  // Set default target language from agent
  useEffect(() => {
    if (selectedAgent?.target_language) {
      setTargetLanguage(selectedAgent.target_language)
    }
  }, [selectedAgent])

  // Poll for job updates
  useEffect(() => {
    if (!selectedAgent) return

    const fetchJobs = () => {
      getTranslationJobs(selectedAgent.id)
        .then(res => {
          const jobsData = res.data || []
          setJobs(jobsData)

          // Check if any jobs are active
          const hasActiveJobs = jobsData.some((job: TranslationJob) =>
            ['pending', 'extracting', 'translating', 'generating'].includes(job.status)
          )

          // Poll faster if active, slower if idle
          if (hasActiveJobs) {
            if (jobsPollInterval.current) clearInterval(jobsPollInterval.current)
            jobsPollInterval.current = window.setInterval(fetchJobs, 2000)
          } else {
            if (jobsPollInterval.current) clearInterval(jobsPollInterval.current)
            jobsPollInterval.current = window.setInterval(fetchJobs, 10000)
          }
        })
        .catch(() => {})
    }

    fetchJobs()
    jobsPollInterval.current = window.setInterval(fetchJobs, 2000)

    return () => {
      if (jobsPollInterval.current) clearInterval(jobsPollInterval.current)
    }
  }, [selectedAgent])

  const handleTranslate = async () => {
    if (!sourceText.trim() || !selectedAgent || isTranslating) return

    // Check if model is available
    if (selectedAgent.model_available === false) {
      setTranslatedText(
        `Error: The model '${selectedAgent.llm_model}' is not available. ` +
        `Please check your model configuration or contact your administrator.`
      )
      return
    }

    setIsTranslating(true)
    try {
      const res = await translateText(selectedAgent.id, {
        text: sourceText,
        target_language: targetLanguage,
        simplified,
        source_language: 'auto',
      })
      setTranslatedText(res.data.translated_text)
    } catch (err: any) {
      const errorDetail = err.response?.data?.detail || err.message
      // Show user-friendly error message
      setTranslatedText(`Error: ${errorDetail}`)
    } finally {
      setIsTranslating(false)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const handleFileUpload = async () => {
    if (!selectedFile || !selectedAgent || isUploading) return

    // Check if model is available
    if (selectedAgent.model_available === false) {
      alert(
        `The model '${selectedAgent.llm_model}' is not available. ` +
        `Please check your model configuration or contact your administrator.`
      )
      return
    }

    setIsUploading(true)
    try {
      await translateFile(selectedAgent.id, selectedFile, targetLanguage, simplified)
      setSelectedFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (err: any) {
      const errorDetail = err.response?.data?.detail || err.message
      alert(`Upload failed: ${errorDetail}`)
    } finally {
      setIsUploading(false)
    }
  }

  const handleDownload = async (jobId: number, filename: string) => {
    try {
      const res = await downloadTranslation(jobId)
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch {
      // Download failed silently
    }
  }

  const handleCopy = async () => {
    if (translatedText) {
      await navigator.clipboard.writeText(translatedText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleClear = () => {
    setSourceText('')
    setTranslatedText('')
  }

  const getJobStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-emerald-400" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />
      case 'pending':
      case 'extracting':
      case 'translating':
      case 'generating':
        return <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
      default:
        return <AlertCircle className="w-5 h-5 text-yellow-400" />
    }
  }

  const getJobStatusLabel = (status: string, progress: number) => {
    switch (status) {
      case 'pending':
        return 'Waiting to start...'
      case 'extracting':
        return `Extracting text from document... (${progress}%)`
      case 'translating':
        return `Translating content... (${progress}%)`
      case 'generating':
        return `Generating document... (${progress}%)`
      case 'completed':
        return 'Completed'
      case 'failed':
        return 'Failed'
      default:
        return `${status} (${progress}%)`
    }
  }

  const formatProcessingTime = (seconds: number | null): string => {
    if (seconds === null || seconds === undefined) return ''
    if (seconds < 60) return `${Math.round(seconds)}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.round(seconds % 60)
    if (minutes < 60) return `${minutes}m ${remainingSeconds}s`
    const hours = Math.floor(minutes / 60)
    const remainingMinutes = minutes % 60
    return `${hours}h ${remainingMinutes}m`
  }

  const getShortDirectory = (path: string | null): string => {
    if (!path) return ''
    const parts = path.split('/')
    return parts.slice(-2).join('/')
  }

  if (!selectedAgent) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Languages className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">No Agent Selected</h3>
          <p className="text-gray-500">Please select a translation agent from the sidebar</p>
        </div>
      </div>
    )
  }

  const isModelUnavailable = selectedAgent.model_available === false

  return (
    <div className="h-full flex flex-col">
      {/* Warning banner if model is unavailable */}
      {isModelUnavailable && (
        <div className="p-4 bg-red-500/20 border-b border-red-500/30">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-400">
                Model '{selectedAgent.llm_model}' is not available
              </p>
              <p className="text-xs text-red-300/80 mt-1">
                Translation is disabled. Please check your model configuration or contact your administrator.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="p-4 border-b border-white/10 bg-black/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
              <Languages className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="font-semibold text-white">{selectedAgent.name}</h2>
              <p className="text-xs text-gray-500">Translation Agent</p>
            </div>
          </div>

          {/* Translation options */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Target:</span>
              <select
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="px-3 py-2 bg-white/5 border border-white/10 rounded-xl text-white text-sm focus:outline-none focus:border-emerald-500"
              >
                {Object.entries(languages).map(([code, name]) => (
                  <option key={code} value={code}>{name}</option>
                ))}
              </select>
            </div>
            <button
              onClick={() => setSimplified(!simplified)}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-colors ${
                simplified
                  ? 'bg-emerald-500/20 text-emerald-400'
                  : 'bg-white/5 text-gray-400 hover:text-white'
              }`}
            >
              {simplified ? <ToggleRight className="w-5 h-5" /> : <ToggleLeft className="w-5 h-5" />}
              Simplified
            </button>
          </div>
        </div>
      </div>

      {/* Tab selector */}
      <div className="px-4 pt-4 flex gap-2">
        <button
          onClick={() => setActiveTab('text')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-colors ${
            activeTab === 'text'
              ? 'bg-emerald-600 text-white'
              : 'bg-white/5 text-gray-400 hover:text-white'
          }`}
        >
          <Languages className="w-4 h-4" />
          Text Translation
        </button>
        <button
          onClick={() => setActiveTab('file')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl font-medium transition-colors ${
            activeTab === 'file'
              ? 'bg-emerald-600 text-white'
              : 'bg-white/5 text-gray-400 hover:text-white'
          }`}
        >
          <FileText className="w-4 h-4" />
          File Translation
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Text Translation Tab - Side by Side */}
        {activeTab === 'text' && (
          <div className="h-full flex flex-col gap-4">
            <div className="flex-1 grid grid-cols-2 gap-4">
              {/* Source text panel */}
              <div className="flex flex-col bg-white/5 border border-white/10 rounded-2xl overflow-hidden">
                <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-400">Source Text</span>
                  <button
                    onClick={handleClear}
                    disabled={!sourceText}
                    className="p-1.5 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    title="Clear"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <textarea
                  value={sourceText}
                  onChange={(e) => setSourceText(e.target.value)}
                  placeholder="Enter text to translate..."
                  className="flex-1 p-4 bg-transparent text-white placeholder-gray-500 focus:outline-none resize-none text-lg leading-relaxed"
                />
                <div className="px-4 py-3 border-t border-white/10 flex items-center justify-between">
                  <span className="text-xs text-gray-500">
                    {sourceText.length} characters
                  </span>
                </div>
              </div>

              {/* Translated text panel */}
              <div className="flex flex-col bg-white/5 border border-white/10 rounded-2xl overflow-hidden">
                <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
                  <span className="text-sm font-medium text-emerald-400">
                    {languages[targetLanguage] || targetLanguage}
                    {simplified && ' (Simplified)'}
                  </span>
                  <button
                    onClick={handleCopy}
                    disabled={!translatedText}
                    className="p-1.5 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
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
                  {isTranslating ? (
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center">
                        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin mx-auto mb-3" />
                        <p className="text-gray-400 text-sm">Translating...</p>
                      </div>
                    </div>
                  ) : translatedText ? (
                    <p className="text-white text-lg leading-relaxed whitespace-pre-wrap">
                      {translatedText}
                    </p>
                  ) : (
                    <div className="flex items-center justify-center h-full">
                      <p className="text-gray-500 text-center">
                        Translation will appear here
                      </p>
                    </div>
                  )}
                </div>
                <div className="px-4 py-3 border-t border-white/10 flex items-center justify-between">
                  <span className="text-xs text-gray-500">
                    {translatedText.length} characters
                  </span>
                </div>
              </div>
            </div>

            {/* Translate button */}
            <div className="flex justify-center">
                <button
                  onClick={handleTranslate}
                  disabled={!sourceText.trim() || isTranslating || isModelUnavailable}
                  className="flex items-center gap-3 px-8 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-emerald-500/25"
                >
                {isTranslating ? (
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
          <div className="h-full flex flex-col gap-4">
            {/* Upload area */}
            <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Upload Document</h3>
              <div className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center">
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
                        className="px-4 py-2 bg-white/5 text-gray-300 rounded-lg hover:bg-white/10"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleFileUpload}
                        disabled={isUploading || isModelUnavailable}
                        className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-lg disabled:opacity-50"
                      >
                        {isUploading ? (
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
                  <div>
                    <Upload className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-400 mb-2">
                      Drag and drop a file, or{' '}
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="text-emerald-400 hover:text-emerald-300"
                      >
                        browse
                      </button>
                    </p>
                    <p className="text-gray-500 text-sm">
                      Supported: PDF, DOCX, PPTX, TXT, Markdown
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Jobs list */}
            <div className="flex-1 bg-white/5 border border-white/10 rounded-2xl p-6 overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Translation Jobs</h3>
                <button
                  onClick={() => {
                    getTranslationJobs(selectedAgent.id)
                      .then(res => setJobs(res.data || []))
                  }}
                  className="p-2 rounded-lg text-gray-500 hover:bg-white/10 hover:text-white"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-3">
                {jobs.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No translation jobs yet. Upload a file to get started.
                  </div>
                )}

                {jobs.map((job) => (
                  <div
                    key={job.id}
                    className="bg-black/20 rounded-xl p-4"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {getJobStatusIcon(job.status)}
                        <div>
                          <p className="text-white font-medium">{job.source_filename}</p>
                          <p className="text-gray-400 text-sm">
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
                        <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                          <span>{getJobStatusLabel(job.status, job.progress)}</span>
                          <span>{job.progress}%</span>
                        </div>
                        <div className="w-full bg-white/10 rounded-full h-2">
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
                      <div className="mt-3 pt-3 border-t border-white/10 flex flex-wrap gap-4 text-xs text-gray-500">
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
    </div>
  )
}


import { useState, useRef, useEffect } from 'react'
import {
  Video, Loader2, Upload, FileText, Download,
  CheckCircle2, AlertCircle
} from 'lucide-react'
import { useStore } from '../store'
import { transcribeVideo, addSubtitlesToVideo, getAgents } from '../api'

interface TranscriptionResult {
  transcript: string
  segments: Array<{
    start: number
    end: number
    text: string
  }>
  language: string
  language_probability?: number
}

export default function VideoTranscriptionView() {
  const { selectedAgent } = useStore()
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState<string>('')
  const [subtitleLanguage, setSubtitleLanguage] = useState<string>('')
  const [translationAgents, setTranslationAgents] = useState<any[]>([])
  const [selectedTranslationAgentId, setSelectedTranslationAgentId] = useState<number | null>(null)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isAddingSubtitles, setIsAddingSubtitles] = useState(false)
  const [transcription, setTranscription] = useState<TranscriptionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch translation agents on mount
  useEffect(() => {
    const fetchTranslationAgents = async () => {
      try {
        const response = await getAgents()
        const translationAgentsList = response.data.filter((agent: any) => agent.agent_type === 'translation')
        setTranslationAgents(translationAgentsList)
        // Auto-select first agent if available
        if (translationAgentsList.length > 0) {
          setSelectedTranslationAgentId(translationAgentsList[0].id)
        }
      } catch (err) {
        console.error('Failed to fetch translation agents:', err)
      }
    }
    fetchTranslationAgents()
  }, [])

  if (!selectedAgent) {
    return null
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      // Validate file type
      const validTypes = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo', 'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/webm']
      if (!validTypes.includes(selectedFile.type)) {
        setError('Please select a valid video or audio file (MP4, MP3, WAV, etc.)')
        return
      }
      setFile(selectedFile)
      setError(null)
      setTranscription(null)
      setDownloadUrl(null)
    }
  }

  const handleTranscribe = async () => {
    if (!file || !selectedAgent) {
      setError('Please select a file first')
      return
    }

    setIsTranscribing(true)
    setError(null)
    setTranscription(null)

    try {
      const response = await transcribeVideo(
        selectedAgent.id,
        file,
        language || undefined
      )
      setTranscription(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Transcription failed')
    } finally {
      setIsTranscribing(false)
    }
  }

  const handleAddSubtitles = async () => {
    if (!file || !selectedAgent) {
      setError('Please select a file first')
      return
    }

    setIsAddingSubtitles(true)
    setError(null)
    setDownloadUrl(null)

    try {
      const response = await addSubtitlesToVideo(
        selectedAgent.id,
        file,
        language || undefined,
        subtitleLanguage || undefined,
        selectedTranslationAgentId || undefined
      )

      // Create download URL from blob
      const url = window.URL.createObjectURL(response.data)
      setDownloadUrl(url)

      // Also get transcription for display
      const transcribeResponse = await transcribeVideo(
        selectedAgent.id,
        file,
        language || undefined
      )
      setTranscription(transcribeResponse.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to add subtitles')
    } finally {
      setIsAddingSubtitles(false)
    }
  }

  const handleDownload = () => {
    if (downloadUrl) {
      const a = document.createElement('a')
      a.href = downloadUrl
      a.download = file?.name.replace(/\.[^/.]+$/, '') + '_with_subtitles.mp4' || 'video_with_subtitles.mp4'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    }
  }

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="border-b border-white/10 bg-black/20 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center">
                <Video className="w-6 h-6 text-violet-400" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-white">{selectedAgent.name}</h1>
                <p className="text-sm text-slate-400">Video Transcription</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
          {/* File Upload */}
          <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Upload Video/Audio</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Select File
                </label>
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
                    Source Language (optional - leave empty for auto-detect)
                  </label>
                  <input
                    type="text"
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    placeholder="e.g., en, fr, es"
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Language spoken in the video. Leave empty to auto-detect.
                  </p>
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
                  <p className="text-xs text-slate-500 mt-1">
                    Language for subtitles. Leave empty to use source language. If different, subtitles will be translated.
                  </p>
                </div>
              </div>

              {subtitleLanguage && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Translation Agent (required for translation)
                  </label>
                  <select
                    value={selectedTranslationAgentId || ''}
                    onChange={(e) => setSelectedTranslationAgentId(e.target.value ? parseInt(e.target.value) : null)}
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                  >
                    <option value="">-- Select Translation Agent --</option>
                    {translationAgents.map((agent: any) => (
                      <option key={agent.id} value={agent.id}>
                        {agent.name} {agent.llm_model ? `(${agent.llm_model})` : ''}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-slate-500 mt-1">
                    {translationAgents.length === 0 
                      ? 'No translation agents available. Create a translation agent to enable subtitle translation.'
                      : 'Select a translation agent to use for translating subtitles. If not selected, translation will be skipped.'}
                  </p>
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
                      Transcribing...
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
                      Processing...
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

          {/* Transcription Results */}
          {transcription && (
            <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white">Transcription</h2>
                {transcription.language && (
                  <div className="flex items-center gap-2 text-sm text-slate-400">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>
                      Language: {transcription.language}
                      {transcription.language_probability && (
                        <span className="ml-1">
                          ({Math.round(transcription.language_probability * 100)}%)
                        </span>
                      )}
                    </span>
                  </div>
                )}
              </div>

              {/* Full Transcript */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-slate-400 mb-2">Full Transcript</h3>
                <div className="bg-slate-900/50 rounded-xl p-4 max-h-96 overflow-y-auto">
                  <p className="text-slate-200 whitespace-pre-wrap leading-relaxed">
                    {transcription.transcript}
                  </p>
                </div>
              </div>

              {/* Segments */}
              {transcription.segments && transcription.segments.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-slate-400 mb-2">Segments</h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {transcription.segments.map((segment, idx) => (
                      <div
                        key={idx}
                        className="bg-slate-900/30 rounded-lg p-3 border border-slate-700/50"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs text-slate-500 font-mono">
                            {formatTime(segment.start)} → {formatTime(segment.end)}
                          </span>
                        </div>
                        <p className="text-slate-200 text-sm">{segment.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Download Video with Subtitles */}
          {downloadUrl && (
            <div className="bg-black/20 backdrop-blur-sm border border-white/10 rounded-2xl p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white mb-1">Video with Subtitles Ready</h2>
                  <p className="text-sm text-slate-400">
                    Your video has been processed with embedded subtitles
                  </p>
                </div>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white font-semibold rounded-xl shadow-lg shadow-green-500/25 transition-all"
                >
                  <Download className="w-5 h-5" />
                  Download Video
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

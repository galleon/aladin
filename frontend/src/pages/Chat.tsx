import { useState, useRef, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { agentsApi, conversationsApi, Message, SourceReference } from '../api/client';
import { Link } from 'react-router-dom';
import {
  Send,
  Loader2,
  ThumbsUp,
  ThumbsDown,
  FileText,
  ChevronDown,
  ChevronUp,
  Sparkles,
  User,
  MessageSquare,
  Volume2,
  Table2,
  Image,
} from 'lucide-react';
import { useVoice } from '../hooks/useVoice';
import VoiceButton from '../components/VoiceButton';

// ── Citation card ──────────────────────────────────────────────────────────────

const CONTENT_TYPE_META: Record<string, { label: string; color: string }> = {
  structured: { label: 'Table',  color: 'bg-amber-500/20 text-amber-300 border-amber-500/30' },
  image:      { label: 'Image',  color: 'bg-sky-500/20   text-sky-300   border-sky-500/30'   },
  text:       { label: 'Text',   color: 'bg-violet-500/20 text-violet-300 border-violet-500/30' },
};

/**
 * Renders a mini page thumbnail fetched from the backend with a coloured
 * bbox overlay highlighting the chunk's location on the page.
 *
 * The page image is fetched lazily (only when the component mounts) and
 * cached by the browser via the Cache-Control header set by the API.
 */
function BboxIndicator({
  documentId, pageNo, loc, pageWidth, pageHeight, textType,
}: {
  documentId: number;
  pageNo: number;
  loc: [number, number, number, number];
  pageWidth: number | null;
  pageHeight: number | null;
  textType: string | null;
}) {
  const [pageImgSrc, setPageImgSrc] = useState<string | null>(null);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    fetch(`/api/documents/${documentId}/pages/${pageNo}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
      .then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
      .then((blob) => setPageImgSrc(URL.createObjectURL(blob)))
      .catch(() => { /* silently skip — no thumbnail available */ });
  }, [documentId, pageNo]);

  const [l, t, r, b] = loc;
  const pw = pageWidth  ?? 595;  // A4 default (points)
  const ph = pageHeight ?? 842;
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const x1 = clamp(Math.min(l, r) / pw);
  const x2 = clamp(Math.max(l, r) / pw);
  const y1 = clamp(1 - Math.max(t, b) / ph);  // flip y: Docling bottom-left → CSS top-left
  const y2 = clamp(1 - Math.min(t, b) / ph);
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  // Don't render the placeholder rectangle until we know whether a page image exists
  if (!pageImgSrc) return null;

  return (
    <div className="mt-2 space-y-1">
      <div
        className="relative rounded overflow-hidden border border-slate-700/40"
        style={{ width: 120, height: Math.round(120 * (ph / pw)) }}
        title={`p.${pageNo} · ${textType ?? ''} · x ${pct(x1)}–${pct(x2)} y ${pct(y1)}–${pct(y2)}`}
      >
        <img src={pageImgSrc} className="absolute inset-0 w-full h-full object-cover" alt="" />
        <div
          className="absolute border-2 border-violet-400 bg-violet-400/25 rounded-sm"
          style={{ left: pct(x1), top: pct(y1), width: pct(x2 - x1), height: pct(y2 - y1) }}
        />
      </div>
    </div>
  );
}

function CitationCard({ source }: { source: SourceReference }) {
  const ct = source.content_type ?? 'text';
  const meta = CONTENT_TYPE_META[ct] ?? CONTENT_TYPE_META.text;
  const Icon = ct === 'structured' ? Table2 : ct === 'image' ? Image : FileText;
  const hasBbox = source.text_location?.length === 4 && source.page != null;

  return (
    <div className="p-3 bg-slate-800/30 rounded-xl border border-slate-700/30">
      {/* Header row */}
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <Icon className="w-4 h-4 text-violet-400 shrink-0" />
        <span className="text-sm font-medium text-slate-300 truncate max-w-[180px]" title={source.filename}>
          {source.filename}
        </span>

        {/* Content-type badge (only when not plain text) */}
        {ct !== 'text' && (
          <span className={`text-xs px-1.5 py-0.5 rounded border font-medium ${meta.color}`}>
            {meta.label}
          </span>
        )}

        {/* Page */}
        {source.page != null && (
          <span className="text-xs text-slate-500">p.{source.page}</span>
        )}

        {/* Score */}
        <span className="text-xs text-slate-500 ml-auto shrink-0">
          {(source.score * 100).toFixed(1)}%
        </span>
      </div>

      {/* For image chunks: render the actual extracted image */}
      {ct === 'image' && source.image_data ? (
        <img
          src={`data:image/jpeg;base64,${source.image_data}`}
          alt={source.chunk_text}
          className="rounded max-w-full max-h-48 object-contain mb-2"
        />
      ) : (
        /* Chunk text preview for text / table chunks */
        <p className="text-sm text-slate-400 line-clamp-3">{source.chunk_text}</p>
      )}

      {/* Bounding box thumbnail (text / table chunks with bbox from rich processor) */}
      {ct !== 'image' && hasBbox && (
        <BboxIndicator
          documentId={source.document_id}
          pageNo={source.page!}
          loc={source.text_location as [number, number, number, number]}
          pageWidth={source.page_width}
          pageHeight={source.page_height}
          textType={source.text_type}
        />
      )}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function Chat() {
  const { agentId, conversationId } = useParams<{ agentId: string; conversationId?: string }>();
  const [message, setMessage] = useState('');
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(
    conversationId ? parseInt(conversationId) : null
  );
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  // Voice functionality
  const {
    recordingState,
    audioLevel,
    isPlaying,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  } = useVoice({
    onTranscript: (text) => {
      setMessage(text);
      setVoiceError(null);
    },
    onError: (error) => {
      setVoiceError(error);
      setTimeout(() => setVoiceError(null), 5000);
    },
  });

  // Fetch agent
  const { data: agent } = useQuery({
    queryKey: ['agent', agentId],
    queryFn: () => agentsApi.get(parseInt(agentId!)).then(res => res.data),
    enabled: !!agentId,
  });

  // Fetch conversation
  const { data: conversation, refetch: refetchConversation } = useQuery({
    queryKey: ['conversation', currentConversationId],
    queryFn: () => conversationsApi.get(currentConversationId!).then(res => res.data),
    enabled: !!currentConversationId,
  });

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      if (currentConversationId) {
        return conversationsApi.chat(currentConversationId, content).then(res => res.data);
      } else {
        return conversationsApi.quickChat(parseInt(agentId!), content).then(res => res.data);
      }
    },
    onSuccess: (data) => {
      if (!currentConversationId) {
        setCurrentConversationId(data.conversation_id);
      }
      refetchConversation();
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });

  // Feedback mutation
  const feedbackMutation = useMutation({
    mutationFn: ({ messageId, thumbsUp }: { messageId: number; thumbsUp: boolean | null }) =>
      conversationsApi.addFeedback(messageId, thumbsUp).then(res => res.data),
    onSuccess: () => {
      refetchConversation();
    },
  });

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim() || sendMessageMutation.isPending) return;

    const content = message;
    setMessage('');
    sendMessageMutation.mutate(content);
  };

  const toggleSources = (messageId: number) => {
    setExpandedSources((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  const handleFeedback = (messageId: number, thumbsUp: boolean) => {
    feedbackMutation.mutate({ messageId, thumbsUp });
  };

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 pb-6 border-b border-slate-800/50">
        <div className="flex items-center gap-4">
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20">
            <Sparkles className="w-6 h-6 text-violet-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">{agent?.name || 'Chat'}</h1>
            <p className="text-sm text-slate-400">{agent?.llm_model}</p>
          </div>
        </div>
        {agentId && (
          <Link
            to={`/conversations?agent_id=${agentId}`}
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-violet-400 transition-colors"
          >
            <MessageSquare className="w-4 h-4" />
            Conversations
          </Link>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6">
        {conversation?.messages?.map((msg: Message) => (
          <div
            key={msg.id}
            className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
          >
            {/* Avatar */}
            <div
              className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${
                msg.role === 'user'
                  ? 'bg-slate-700'
                  : 'bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20'
              }`}
            >
              {msg.role === 'user' ? (
                <User className="w-5 h-5 text-slate-300" />
              ) : (
                <Sparkles className="w-5 h-5 text-violet-400" />
              )}
            </div>

            {/* Message content */}
            <div className={`flex-1 max-w-[80%] ${msg.role === 'user' ? 'text-right' : ''}`}>
              <div
                className={`inline-block rounded-2xl px-5 py-3 ${
                  msg.role === 'user'
                    ? 'bg-violet-600 text-white'
                    : 'bg-slate-800/50 text-slate-200 border border-slate-700/50'
                }`}
              >
                <p className="whitespace-pre-wrap text-left">{msg.content}</p>
              </div>

              {/* Sources for assistant messages */}
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="mt-3">
                  <button
                    onClick={() => toggleSources(msg.id)}
                    className="flex items-center gap-2 text-sm text-slate-400 hover:text-violet-400 transition-colors"
                  >
                    <FileText className="w-4 h-4" />
                    {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
                    {expandedSources.has(msg.id) ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>

                  {expandedSources.has(msg.id) && (
                    <div className="mt-2 space-y-2">
                      {msg.sources.map((source: SourceReference, idx: number) => (
                        <CitationCard key={idx} source={source} />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Feedback for assistant messages */}
              {msg.role === 'assistant' && (
                <div className="mt-3 flex items-center gap-2">
                  {/* TTS button */}
                  <button
                    onClick={() => speak(msg.content)}
                    disabled={isPlaying}
                    className="p-2 rounded-lg transition-colors text-slate-500 hover:bg-slate-800 hover:text-violet-400 disabled:opacity-50"
                    title="Read aloud"
                  >
                    <Volume2 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleFeedback(msg.id, true)}
                    className={`p-2 rounded-lg transition-colors ${
                      msg.feedback?.thumbs_up === true
                        ? 'bg-green-500/20 text-green-400'
                        : 'text-slate-500 hover:bg-slate-800 hover:text-green-400'
                    }`}
                    title="Helpful"
                  >
                    <ThumbsUp className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleFeedback(msg.id, false)}
                    className={`p-2 rounded-lg transition-colors ${
                      msg.feedback?.thumbs_up === false
                        ? 'bg-red-500/20 text-red-400'
                        : 'text-slate-500 hover:bg-slate-800 hover:text-red-400'
                    }`}
                    title="Not helpful"
                  >
                    <ThumbsDown className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {sendMessageMutation.isPending && (
          <div className="flex gap-4">
            <div className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20">
              <Sparkles className="w-5 h-5 text-violet-400" />
            </div>
            <div className="flex items-center gap-2 px-5 py-3 bg-slate-800/50 rounded-2xl border border-slate-700/50">
              <Loader2 className="w-4 h-4 text-violet-400 animate-spin" />
              <span className="text-slate-400">Thinking...</span>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!conversation?.messages?.length && !sendMessageMutation.isPending && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20">
              <MessageSquare className="w-8 h-8 text-violet-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Start a conversation</h3>
            <p className="text-slate-400 max-w-md">
              Ask {agent?.name || 'the agent'} anything about your documents. It will search through
              your data domain and provide relevant answers.
            </p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSendMessage} className="pt-4 border-t border-slate-800/50">
        {/* Voice error notification */}
        {voiceError && (
          <div className="mb-3 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
            {voiceError}
          </div>
        )}
        
        <div className="flex gap-3">
          {/* Voice button */}
          <VoiceButton
            recordingState={recordingState}
            audioLevel={audioLevel}
            isPlaying={isPlaying}
            onStartRecording={startRecording}
            onStopRecording={stopRecording}
            onStopSpeaking={stopSpeaking}
            disabled={sendMessageMutation.isPending}
          />
          
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message or use voice..."
            className="flex-1 px-5 py-4 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 transition-all"
            disabled={sendMessageMutation.isPending || recordingState !== 'idle'}
          />
          <button
            type="submit"
            disabled={!message.trim() || sendMessageMutation.isPending || recordingState !== 'idle'}
            className="px-6 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {sendMessageMutation.isPending ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </form>
    </div>
  );
}


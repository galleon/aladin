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
} from 'lucide-react';
import { useVoice } from '../hooks/useVoice';
import VoiceButton from '../components/VoiceButton';

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
                        <div
                          key={idx}
                          className="p-3 bg-slate-800/30 rounded-xl border border-slate-700/30"
                        >
                          <div className="flex items-center gap-2 mb-2">
                            <FileText className="w-4 h-4 text-violet-400" />
                            <span className="text-sm font-medium text-slate-300">
                              {source.filename}
                            </span>
                            {source.page && (
                              <span className="text-xs text-slate-500">Page {source.page}</span>
                            )}
                            <span className="text-xs text-slate-500 ml-auto">
                              Score: {(source.score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <p className="text-sm text-slate-400 line-clamp-3">{source.chunk_text}</p>
                        </div>
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


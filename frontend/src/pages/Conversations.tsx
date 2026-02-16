import { Link, useSearchParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { conversationsApi } from '../api/client';
import { MessageSquare, Trash2, Loader2, Clock } from 'lucide-react';

export default function Conversations() {
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const agentIdParam = searchParams.get('agent_id');
  const agentId = agentIdParam ? parseInt(agentIdParam, 10) : undefined;
  const filterByAgent = agentId !== undefined && !Number.isNaN(agentId);

  const { data: conversations, isLoading } = useQuery({
    queryKey: ['conversations', filterByAgent ? agentId : null],
    queryFn: () => conversationsApi.list(filterByAgent ? agentId : undefined).then(res => res.data),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => conversationsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (days === 1) {
      return 'Yesterday';
    } else if (days < 7) {
      return date.toLocaleDateString([], { weekday: 'long' });
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Conversations</h1>
        <p className="text-slate-400">Your chat history</p>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-violet-400 animate-spin" />
        </div>
      )}

      {/* Conversations list */}
      {conversations && conversations.length > 0 && (
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl overflow-hidden">
          <div className="divide-y divide-slate-800/50">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className="p-4 flex items-center gap-4 hover:bg-slate-800/30 transition-colors group"
              >
                <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-violet-500/10 shrink-0">
                  <MessageSquare className="w-6 h-6 text-violet-400" />
                </div>
                <Link
                  to={`/chat/${conv.agent_id}/${conv.id}`}
                  className="flex-1 min-w-0"
                >
                  <h3 className="text-white font-medium truncate hover:text-violet-300 transition-colors">
                    {conv.title || 'Untitled conversation'}
                  </h3>
                  <div className="flex items-center gap-3 text-sm text-slate-400">
                    <span>{conv.agent_name}</span>
                    <span>•</span>
                    <span>{conv.message_count} messages</span>
                  </div>
                </Link>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 text-sm text-slate-500">
                    <Clock className="w-4 h-4" />
                    {formatDate(conv.updated_at)}
                  </div>
                  <button
                    onClick={() => {
                      if (confirm('Delete this conversation?')) {
                        deleteMutation.mutate(conv.id);
                      }
                    }}
                    className="p-2 rounded-lg text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                    title="Delete"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {conversations && conversations.length === 0 && (
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-12 text-center">
          <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-2xl bg-violet-500/10">
            <MessageSquare className="w-8 h-8 text-violet-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">No conversations yet</h3>
          <p className="text-slate-400 mb-6">
            Start chatting with an agent to see your conversations here.
          </p>
          <Link
            to="/agents"
            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all"
          >
            View Agents
          </Link>
        </div>
      )}
    </div>
  );
}


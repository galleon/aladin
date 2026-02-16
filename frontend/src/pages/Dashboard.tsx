import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { dataDomainsApi, agentsApi, statsApi } from '../api/client';
import { Database, Bot, MessageSquare, Plus, ArrowRight, Sparkles, Hash, Video, Languages } from 'lucide-react';

export default function Dashboard() {
  const { data: dataDomains } = useQuery({
    queryKey: ['dataDomains'],
    queryFn: () => dataDomainsApi.list().then(res => res.data),
  });

  const { data: agents } = useQuery({
    queryKey: ['agents'],
    queryFn: () => agentsApi.list().then(res => res.data),
  });

  const { data: usage } = useQuery({
    queryKey: ['usageStats'],
    queryFn: async () => {
      try {
        const res = await statsApi.getUsage();
        const d = res?.data;
        return {
          prompts_count: typeof d?.prompts_count === 'number' ? d.prompts_count : 0,
          input_tokens: typeof d?.input_tokens === 'number' ? d.input_tokens : 0,
          output_tokens: typeof d?.output_tokens === 'number' ? d.output_tokens : 0,
          total_tokens: typeof d?.total_tokens === 'number' ? d.total_tokens : (typeof d?.input_tokens === 'number' && typeof d?.output_tokens === 'number' ? d.input_tokens + d.output_tokens : 0),
        };
      } catch {
        return { prompts_count: 0, input_tokens: 0, output_tokens: 0, total_tokens: 0 };
      }
    },
    retry: false,
    staleTime: 60_000,
  });

  const stats = [
    {
      label: 'Data Domains',
      value: dataDomains?.length || 0,
      icon: Database,
      iconBg: 'bg-violet-500/10',
      iconColor: 'text-violet-400',
      link: '/data-domains',
    },
    {
      label: 'Agents',
      value: agents?.length || 0,
      icon: Bot,
      iconBg: 'bg-fuchsia-500/10',
      iconColor: 'text-fuchsia-400',
      link: '/agents',
    },
    {
      label: 'Prompts',
      value: usage?.prompts_count ?? '–',
      icon: MessageSquare,
      iconBg: 'bg-cyan-500/10',
      iconColor: 'text-cyan-400',
      link: null,
    },
    {
      label: 'Tokens',
      value: usage?.total_tokens != null ? usage.total_tokens.toLocaleString() : '–',
      icon: Hash,
      iconBg: 'bg-amber-500/10',
      iconColor: 'text-amber-400',
      link: null,
    },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
        <p className="text-slate-400">Welcome to ALADIN</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const cardClassName = "group bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6 hover:border-violet-500/30 transition-all duration-300";
          const inner = (
            <>
              <div className="flex items-center justify-between mb-4">
                <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${stat.iconBg}`}>
                  <stat.icon className={`w-6 h-6 ${stat.iconColor}`} />
                </div>
                {stat.link != null && <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-violet-400 transition-colors" />}
              </div>
              <p className="text-3xl font-bold text-white mb-1">{stat.value}</p>
              <p className="text-slate-400">{stat.label}</p>
            </>
          );
          return stat.link != null ? (
            <Link key={stat.label} to={stat.link} className={cardClassName}>{inner}</Link>
          ) : (
            <div key={stat.label} className={cardClassName}>{inner}</div>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Link
            to="/data-domains"
            className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700/50 hover:border-violet-500/30 transition-all group"
          >
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-violet-500/10">
              <Plus className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <p className="font-medium text-white group-hover:text-violet-300 transition-colors">
                Create Data Domain
              </p>
              <p className="text-sm text-slate-500">Upload and index documents</p>
            </div>
          </Link>

          <Link
            to="/agents"
            className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700/50 hover:border-fuchsia-500/30 transition-all group"
          >
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-fuchsia-500/10">
              <Bot className="w-5 h-5 text-fuchsia-400" />
            </div>
            <div>
              <p className="font-medium text-white group-hover:text-fuchsia-300 transition-colors">
                Create Agent
              </p>
              <p className="text-sm text-slate-500">Configure a RAG agent</p>
            </div>
          </Link>
        </div>
      </div>

      {/* Recent Agents */}
      {agents && agents.length > 0 && (
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Your Agents</h2>
            <Link to="/agents" className="text-violet-400 hover:text-violet-300 text-sm font-medium">
              View all
            </Link>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.slice(0, 6).map((agent) => {
              const isVideo = agent.agent_type === 'video_transcription';
              const isTranslation = agent.agent_type === 'translation';
              const AgentIcon = isVideo ? Video : isTranslation ? Languages : Sparkles;
              return (
              <a
                key={agent.id}
                href={`${import.meta.env.VITE_CHAT_UI_URL || 'http://localhost:7860'}?agent_id=${agent.id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/30 hover:border-violet-500/30 transition-all group"
              >
                <div className={`flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br shrink-0 ${isVideo ? 'from-blue-500/20 to-cyan-500/20' : isTranslation ? 'from-emerald-500/20 to-teal-500/20' : 'from-violet-500/20 to-fuchsia-500/20'}`}>
                  <AgentIcon className={`w-5 h-5 ${isVideo ? 'text-blue-400' : isTranslation ? 'text-emerald-400' : 'text-violet-400'}`} />
                </div>
                <div className="min-w-0">
                  <p className="font-medium text-white truncate group-hover:text-violet-300 transition-colors">
                    {agent.name}
                  </p>
                  <p className="text-sm text-slate-500 truncate">{agent.llm_model}</p>
                </div>
              </a>
            );
            })}
          </div>
        </div>
      )}

      {/* Empty State */}
      {(!agents || agents.length === 0) && (!dataDomains || dataDomains.length === 0) && (
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800/50 rounded-2xl p-12 text-center">
          <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20">
            <Sparkles className="w-8 h-8 text-violet-400" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Get Started</h3>
          <p className="text-slate-400 mb-6 max-w-md mx-auto">
            Create your first data domain by uploading documents, then create an agent to chat with your data.
          </p>
          <Link
            to="/data-domains"
            className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-semibold rounded-xl shadow-lg shadow-violet-500/25 transition-all"
          >
            Create Data Domain
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      )}
    </div>
  );
}

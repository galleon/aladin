import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { authApi, User } from './api/client';
import Layout from './components/Layout';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import DataDomains from './pages/DataDomains';
import DataDomainDetail from './pages/DataDomainDetail';
import DataDomainInspect from './pages/DataDomainInspect';
import Agents from './pages/Agents';
import AgentDetail from './pages/AgentDetail';
import TranscriptionJobs from './pages/TranscriptionJobs';
import Chat from './pages/Chat';
import TranslationChat from './pages/TranslationChat';
import Conversations from './pages/Conversations';

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 5 * 60 * 1000,
            retry: 1,
        },
    },
});

// Auth Context
interface AuthContextType {
    user: User | null;
    isLoading: boolean;
    login: (token: string) => Promise<void>;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}

function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem('access_token');
        if (!token) {
            setIsLoading(false);
            return;
        }
        // Timeout so we don't hang forever if backend is unreachable (user can still open login)
        const authTimeoutMs = 8000;
        const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('auth_timeout')), authTimeoutMs)
        );
        Promise.race([authApi.getCurrentUser(), timeoutPromise])
            .then(res => setUser((res as { data: User }).data))
            .catch(() => localStorage.removeItem('access_token'))
            .finally(() => setIsLoading(false));
    }, []);

    const login = async (token: string) => {
        localStorage.setItem('access_token', token);
        const res = await authApi.getCurrentUser();
        setUser(res.data);
    };

    const logout = () => {
        localStorage.removeItem('access_token');
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, isLoading, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

// Protected Route
function ProtectedRoute({ children }: { children: ReactNode }) {
    const { user, isLoading } = useAuth();

    if (isLoading) {
        return (
            <div className="min-h-screen bg-slate-950 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-violet-500"></div>
            </div>
        );
    }

    if (!user) {
        return <Navigate to="/login" />;
    }

    return <>{children}</>;
}

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <AuthProvider>
                <Router>
                    <Routes>
                        <Route path="/login" element={<Login />} />
                        <Route path="/register" element={<Register />} />
                        <Route
                            path="/*"
                            element={
                                <ProtectedRoute>
                                    <Layout>
                                        <Routes>
                                            <Route path="/" element={<Dashboard />} />
                                            <Route path="/data-domains" element={<DataDomains />} />
                                            <Route path="/data-domains/:id" element={<DataDomainDetail />} />
                                            <Route path="/data-domains/:id/inspect" element={<DataDomainInspect />} />
                                            <Route path="/agents" element={<Agents />} />
                                            <Route path="/agents/:id" element={<AgentDetail />} />
                                            <Route path="/jobs" element={<TranscriptionJobs />} />
                                            <Route path="/transcription-jobs" element={<Navigate to="/jobs" replace />} />
                                            <Route path="/conversations" element={<Conversations />} />
                                            <Route path="/chat/:agentId" element={<Chat />} />
                                            <Route path="/chat/:agentId/:conversationId" element={<Chat />} />
                                            <Route path="/translate/:agentId" element={<TranslationChat />} />
                                        </Routes>
                                    </Layout>
                                </ProtectedRoute>
                            }
                        />
                    </Routes>
                </Router>
            </AuthProvider>
        </QueryClientProvider>
    );
}

export default App;

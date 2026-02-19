/**
 * AvatarView — Live Avatar Agent UI
 *
 * When the selected agent has type 'avatar' this view:
 *   1. Calls POST /agents/{id}/session to get a LiveKit room token.
 *   2. Renders a @livekit/components-react <VideoConference> so the user can
 *      see the avatar video stream.
 *   3. Keeps the text chat visible as subtitles overlaid on the video.
 *
 * Prerequisites:
 *   npm install @livekit/components-react @livekit/client
 */

import { useState, useRef } from 'react'
import { Bot, Loader2, AlertCircle, Send } from 'lucide-react'
import { createAvatarSession } from '../api'

// LiveKit imports — gracefully degrade when package is not installed.
let LiveKitRoom: any = null
let VideoConference: any = null
let RoomAudioRenderer: any = null
let useRoomContext: (() => any) | null = null

try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const lkComponents = require('@livekit/components-react')
    LiveKitRoom = lkComponents.LiveKitRoom
    VideoConference = lkComponents.VideoConference
    RoomAudioRenderer = lkComponents.RoomAudioRenderer
    useRoomContext = lkComponents.useRoomContext
} catch {
    // @livekit/components-react not yet installed — avatar video is disabled,
    // only the session start screen is shown.
}

interface AvatarAgentType {
    id: number
    name: string
    description?: string
    llm_model: string | null
    agent_type?: string
    avatar_config?: {
        video_source_url?: string
        image_url?: string
    } | null
}

interface AvatarViewProps {
    agent: AvatarAgentType
}

interface SubtitleLine {
    id: number
    text: string
}

// ---------------------------------------------------------------------------
// Inner component — rendered inside <LiveKitRoom> so useRoomContext() works
// ---------------------------------------------------------------------------
function AvatarRoomContent({
    subtitles,
    onSubtitle,
}: {
    subtitles: SubtitleLine[]
    onSubtitle: (line: SubtitleLine) => void
}) {
    const room = useRoomContext ? useRoomContext() : null
    const [inputText, setInputText] = useState('')
    const subtitleIdRef = useRef(0)

    // Receive subtitle data messages from the avatar worker
    if (room) {
        room.on('dataReceived', (payload: Uint8Array) => {
            try {
                const text = new TextDecoder().decode(payload)
                if (text) {
                    onSubtitle({ id: ++subtitleIdRef.current, text })
                }
            } catch {
                // ignore decode errors
            }
        })
    }

    const sendMessage = () => {
        if (!inputText.trim() || !room) return
        const encoder = new TextEncoder()
        room.localParticipant?.publishData(encoder.encode(inputText.trim()), { reliable: true })
        setInputText('')
    }

    return (
        <>
            {VideoConference && <VideoConference className="w-full h-full" />}
            {RoomAudioRenderer && <RoomAudioRenderer />}

            {/* Subtitle overlay */}
            {subtitles.length > 0 && (
                <div className="absolute bottom-20 left-0 right-0 flex flex-col items-center gap-1 pointer-events-none px-4">
                    {subtitles.map(line => (
                        <div
                            key={line.id}
                            className="bg-black/70 text-white text-sm px-3 py-1 rounded max-w-2xl text-center"
                        >
                            {line.text}
                        </div>
                    ))}
                </div>
            )}

            {/* Text input bar */}
            <div className="absolute bottom-0 left-0 right-0 p-3 bg-black/40 backdrop-blur flex gap-2">
                <input
                    type="text"
                    value={inputText}
                    onChange={e => setInputText(e.target.value)}
                    onKeyDown={e => {
                        if (e.key === 'Enter') sendMessage()
                    }}
                    placeholder="Type a message to the avatar…"
                    className="flex-1 bg-white/10 border border-white/20 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-violet-400"
                />
                <button
                    onClick={sendMessage}
                    className="p-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg transition-colors"
                >
                    <Send className="w-4 h-4" />
                </button>
            </div>
        </>
    )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function AvatarView({ agent }: AvatarViewProps) {
    const [sessionToken, setSessionToken] = useState<string | null>(null)
    const [roomName, setRoomName] = useState<string>('')
    const [livekitUrl, setLivekitUrl] = useState<string>('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [subtitles, setSubtitles] = useState<SubtitleLine[]>([])

    const startSession = async () => {
        setLoading(true)
        setError(null)
        try {
            const res = await createAvatarSession(agent.id)
            setSessionToken(res.data.token)
            setRoomName(res.data.room_name)
            setLivekitUrl(res.data.livekit_url)
        } catch (err: any) {
            setError(err.response?.data?.detail || err.message || 'Failed to start session')
        } finally {
            setLoading(false)
        }
    }

    const addSubtitle = (line: SubtitleLine) => {
        setSubtitles(prev => [...prev.slice(-4), line])
    }

    if (!sessionToken) {
        return (
            <div className="h-screen flex flex-col items-center justify-center gap-6 bg-gray-950">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                    <Bot className="w-8 h-8 text-white" />
                </div>
                <div className="text-center">
                    <h2 className="text-xl font-semibold text-white mb-1">{agent.name}</h2>
                    <p className="text-sm text-gray-400">{agent.description || 'Live Avatar Agent'}</p>
                </div>
                {error && (
                    <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 px-4 py-2 rounded-lg">
                        <AlertCircle className="w-4 h-4 flex-shrink-0" />
                        {error}
                    </div>
                )}
                <button
                    onClick={startSession}
                    disabled={loading}
                    className="px-6 py-3 bg-violet-600 hover:bg-violet-500 disabled:opacity-60 text-white rounded-xl font-medium flex items-center gap-2 transition-colors"
                >
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Bot className="w-4 h-4" />}
                    {loading ? 'Starting session…' : 'Start Avatar Session'}
                </button>
            </div>
        )
    }

    // LiveKit not installed — show a clear fallback message
    if (!LiveKitRoom) {
        return (
            <div className="h-screen flex flex-col items-center justify-center gap-4 bg-gray-950 text-gray-400">
                <AlertCircle className="w-8 h-8" />
                <p className="text-sm text-center max-w-xs">
                    @livekit/components-react is not installed. Run{' '}
                    <code className="text-violet-400">npm install @livekit/components-react @livekit/client</code>{' '}
                    in <em>chat-ui/</em> to enable the live video experience.
                </p>
                <p className="text-xs text-gray-600">Room: {roomName}</p>
            </div>
        )
    }

    return (
        <div className="h-screen flex flex-col bg-gray-950 relative overflow-hidden">
            <LiveKitRoom
                serverUrl={livekitUrl}
                token={sessionToken}
                video={false}
                audio={false}
                className="flex-1 min-h-0 relative"
            >
                <AvatarRoomContent subtitles={subtitles} onSubtitle={addSubtitle} />
            </LiveKitRoom>
        </div>
    )
}


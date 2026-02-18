/**
 * Voice button component with visual feedback for recording state
 */

import { Mic, Square, VolumeX } from 'lucide-react';
import { RecordingState } from '../hooks/useVoice';

interface VoiceButtonProps {
    recordingState: RecordingState;
    audioLevel: number;
    isPlaying: boolean;
    onStartRecording: () => void;
    onStopRecording: () => void;
    onStopSpeaking: () => void;
    disabled?: boolean;
}

export default function VoiceButton({
    recordingState,
    audioLevel,
    isPlaying,
    onStartRecording,
    onStopRecording,
    onStopSpeaking,
    disabled = false,
}: VoiceButtonProps) {
    const handleClick = () => {
        if (isPlaying) {
            onStopSpeaking();
        } else if (recordingState === 'recording') {
            onStopRecording();
        } else if (recordingState === 'idle' && !disabled) {
            onStartRecording();
        }
    };

    const getButtonColor = () => {
        if (recordingState === 'recording') return 'bg-red-600 hover:bg-red-700';
        if (recordingState === 'processing') return 'bg-yellow-600';
        if (isPlaying) return 'bg-green-600 hover:bg-green-700';
        return 'bg-violet-600 hover:bg-violet-700';
    };

    const getIcon = () => {
        if (isPlaying) return <VolumeX className="w-5 h-5" />;
        if (recordingState === 'recording') return <Square className="w-5 h-5" />;
        if (recordingState === 'processing') return <Mic className="w-5 h-5 animate-pulse" />;
        return <Mic className="w-5 h-5" />;
    };

    const getTooltip = () => {
        if (isPlaying) return 'Stop playback';
        if (recordingState === 'recording') return 'Stop recording';
        if (recordingState === 'processing') return 'Processing...';
        return 'Start voice recording';
    };

    return (
        <div className="relative">
            <button
                onClick={handleClick}
                disabled={disabled || recordingState === 'processing'}
                className={`
                    ${getButtonColor()}
                    disabled:bg-slate-700 disabled:cursor-not-allowed
                    text-white rounded-xl p-3 
                    transition-all duration-200
                    hover:shadow-lg
                    focus:outline-none focus:ring-2 focus:ring-violet-500
                    relative overflow-hidden
                `}
                title={getTooltip()}
            >
                {/* Visual feedback for audio level */}
                {recordingState === 'recording' && (
                    <div
                        className="absolute inset-0 bg-white opacity-20 transition-opacity duration-100"
                        style={{
                            opacity: Math.min(audioLevel / 100, 0.5),
                        }}
                    />
                )}
                
                {/* Icon */}
                <div className="relative z-10">
                    {getIcon()}
                </div>
            </button>

            {/* Recording indicator */}
            {recordingState === 'recording' && (
                <div className="absolute -top-1 -right-1 flex items-center justify-center">
                    <span className="relative flex h-3 w-3">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                </div>
            )}

            {/* Audio level indicator */}
            {recordingState === 'recording' && (
                <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-8 h-1 bg-slate-700 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-red-500 transition-all duration-100"
                        style={{
                            width: `${Math.min(audioLevel, 100)}%`,
                        }}
                    />
                </div>
            )}
        </div>
    );
}

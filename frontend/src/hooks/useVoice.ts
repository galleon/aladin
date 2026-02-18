/**
 * Voice chat hook with Voice Activity Detection (VAD)
 *
 * Features:
 * - Record audio with microphone
 * - Simple VAD using audio level detection
 * - Speech-to-Text conversion
 * - Text-to-Speech playback
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { voiceApi } from '../api/client';

interface UseVoiceOptions {
    onTranscript?: (text: string) => void;
    onError?: (error: string) => void;
    vadThreshold?: number; // Audio level threshold for voice detection (0-255)
    vadSilenceDuration?: number; // Milliseconds of silence before stopping
}

export type RecordingState = 'idle' | 'recording' | 'processing';

export function useVoice({
    onTranscript,
    onError,
    vadThreshold = 20,
    vadSilenceDuration = 2000,
}: UseVoiceOptions = {}) {
    const [recordingState, setRecordingState] = useState<RecordingState>('idle');
    const [audioLevel, setAudioLevel] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const audioElementRef = useRef<HTMLAudioElement | null>(null);

    // Clean up audio resources
    const cleanup = useCallback(() => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        analyserRef.current = null;
        mediaRecorderRef.current = null;
        setAudioLevel(0);
    }, []);

    // Stop recording — defined before monitorAudioLevel to avoid circular ref
    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, []);

    // Monitor audio level for VAD
    const monitorAudioLevel = useCallback(() => {
        if (!analyserRef.current) return;

        const analyser = analyserRef.current;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        const checkLevel = () => {
            if (!analyserRef.current) return;

            analyser.getByteFrequencyData(dataArray);

            // Calculate average audio level
            const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
            setAudioLevel(average);

            // VAD: Check if audio level is above threshold
            const isSpeaking = average > vadThreshold;

            if (isSpeaking) {
                // Reset silence timer if speaking
                if (silenceTimerRef.current) {
                    clearTimeout(silenceTimerRef.current);
                    silenceTimerRef.current = null;
                }
            } else if (!silenceTimerRef.current && mediaRecorderRef.current?.state === 'recording') {
                // Start silence timer — use ref check instead of state to avoid stale closure
                silenceTimerRef.current = setTimeout(() => {
                    stopRecording();
                }, vadSilenceDuration);
            }

            animationFrameRef.current = requestAnimationFrame(checkLevel);
        };

        checkLevel();
    }, [vadThreshold, vadSilenceDuration, stopRecording]);

    // Start recording
    const startRecording = useCallback(async () => {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });
            streamRef.current = stream;

            // Set up audio context for level monitoring
            const audioContext = new AudioContext();
            audioContextRef.current = audioContext;
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            analyserRef.current = analyser;

            // Set up media recorder
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4',
            });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                const mimeType = mediaRecorder.mimeType || 'audio/webm';
                const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });

                // Transcribe audio
                setRecordingState('processing');
                try {
                    const response = await voiceApi.transcribe(audioBlob);
                    const text = response.data.text.trim();
                    if (text && onTranscript) {
                        onTranscript(text);
                    }
                } catch (error: any) {
                    const errorMsg = error.response?.data?.detail || error.message || 'Transcription failed';
                    if (onError) {
                        onError(errorMsg);
                    }
                } finally {
                    setRecordingState('idle');
                    cleanup();
                }
            };

            mediaRecorder.start();
            setRecordingState('recording');

            // Start monitoring audio level after recorder is set up
            monitorAudioLevel();
        } catch (error: any) {
            const errorMsg = error.message || 'Failed to access microphone';
            if (onError) {
                onError(errorMsg);
            }
            cleanup();
        }
    }, [monitorAudioLevel, onTranscript, onError, cleanup]);

    // Play text-to-speech
    const speak = useCallback(async (text: string, voice = 'alloy', speed = 1.0) => {
        try {
            setIsPlaying(true);
            const response = await voiceApi.synthesize(text, voice, speed);
            const audioBlob = response.data;
            const audioUrl = URL.createObjectURL(audioBlob);

            // Create and play audio
            const audio = new Audio(audioUrl);
            audioElementRef.current = audio;

            audio.onended = () => {
                setIsPlaying(false);
                URL.revokeObjectURL(audioUrl);
                audioElementRef.current = null;
            };

            audio.onerror = () => {
                setIsPlaying(false);
                URL.revokeObjectURL(audioUrl);
                audioElementRef.current = null;
                if (onError) {
                    onError('Failed to play audio');
                }
            };

            await audio.play();
        } catch (error: any) {
            setIsPlaying(false);
            const errorMsg = error.response?.data?.detail || error.message || 'Speech synthesis failed';
            if (onError) {
                onError(errorMsg);
            }
        }
    }, [onError]);

    // Stop playback
    const stopSpeaking = useCallback(() => {
        if (audioElementRef.current) {
            audioElementRef.current.pause();
            audioElementRef.current.currentTime = 0;
            audioElementRef.current = null;
            setIsPlaying(false);
        }
    }, []);

    // Clean up on unmount
    useEffect(() => {
        return () => {
            cleanup();
            stopSpeaking();
        };
    }, [cleanup, stopSpeaking]);

    return {
        recordingState,
        audioLevel,
        isPlaying,
        startRecording,
        stopRecording,
        speak,
        stopSpeaking,
    };
}

import { useState, useEffect, useRef, useCallback } from "react";
import { Mic, Pause, Square, Play, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { TranscriptionWebSocket, WebSocketMessage } from "@/lib/api";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";

interface DictationInterfaceProps {
  onComplete: (text: string, sessionId: string) => void;
  procedureType: string;
}

const AUDIO_CHUNK_INTERVAL_MS = 2000;

export default function DictationInterface({ onComplete, procedureType }: DictationInterfaceProps) {
  const [transcript, setTranscript] = useState("");
  const [duration, setDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [isFinalizing, setIsFinalizing] = useState(false);

  const wsRef = useRef<TranscriptionWebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket handlers
  const handleWsMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'status':
        if (message.session_id) {
          setSessionId(message.session_id);
        }
        if (message.message?.includes('Processing')) {
          setIsProcessing(true);
          setTimeout(() => setIsProcessing(false), 2000);
        }
        break;

      case 'transcript':
        // Server sends: {type: 'transcript', data: {text: '...', session_id: '...'}}
        const transcriptText = message.data?.text || message.text;
        if (transcriptText) {
          setTranscript(prev => prev + (prev ? ' ' : '') + transcriptText);
          setIsProcessing(true);
          setTimeout(() => setIsProcessing(false), 2000);
        }
        break;

      case 'final':
        // Server sends the complete final transcript
        const finalText = message.data?.text || message.text;
        if (finalText) {
          setTranscript(finalText);
        }
        setIsFinalizing(false);
        if (wsRef.current) {
          wsRef.current.disconnect();
        }
        break;

      case 'error':
        console.error('WebSocket error:', message.error);
        break;
    }
  }, []);

  const handleWsConnect = useCallback(() => {
    setIsConnected(true);
    // Send start message
    wsRef.current?.sendJSON({
      type: 'start',
      session_id: sessionId || undefined,
      chunk_interval_ms: AUDIO_CHUNK_INTERVAL_MS
    });
  }, [sessionId]);

  const handleWsDisconnect = useCallback(() => {
    setIsConnected(false);
  }, []);

  const handleWsError = useCallback((error: string) => {
    console.error('WebSocket error:', error);
  }, []);

  // Audio chunk handler
  const handleAudioChunk = useCallback((blob: Blob) => {
    if (wsRef.current && wsRef.current.isConnected) {
      blob.arrayBuffer().then(buffer => {
        wsRef.current?.send(buffer);
      });
    }
  }, []);

  const { recording, paused, startRecording, pauseRecording, resumeRecording, stopRecording } =
    useAudioRecorder(handleAudioChunk);

  // Duration timer
  useEffect(() => {
    if (recording && !paused) {
      intervalRef.current = setInterval(() => {
        setDuration(d => d + 1);
      }, 1000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [recording, paused]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStart = async () => {
    // Initialize WebSocket
    wsRef.current = new TranscriptionWebSocket(
      handleWsMessage,
      handleWsConnect,
      handleWsDisconnect,
      handleWsError
    );
    wsRef.current.connect();

    // Start recording
    await startRecording();
  };

  const handleStop = () => {
    stopRecording();
    setIsFinalizing(true);

    // Send end message
    wsRef.current?.sendJSON({ type: 'end' });

    // Wait a bit then complete
    setTimeout(() => {
      if (wsRef.current) {
        wsRef.current.disconnect();
      }
      onComplete(transcript, sessionId);
    }, 2000);
  };

  return (
    <div className="space-y-8 max-w-2xl mx-auto">
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-semibold text-primary">
          {recording ? 'Dictation Active' : 'Ready to Record'}
        </h2>
        <p className="text-muted-foreground">
          Recording notes for: <span className="font-medium text-foreground">{procedureType}</span>
        </p>
      </div>

      {/* Visualizer Area */}
      <Card className="p-8 md:p-12 flex flex-col items-center justify-center min-h-[300px] relative overflow-hidden border-2 border-primary/5 shadow-lg bg-white/50 backdrop-blur-sm">

        {/* Animated Waveform Background */}
        <div className="absolute inset-0 flex items-center justify-center opacity-10 pointer-events-none">
          {recording && !paused && (
            <div className="flex gap-1 h-32 items-center">
              {[...Array(20)].map((_, i) => (
                <motion.div
                  key={i}
                  className="w-2 bg-primary rounded-full"
                  animate={{
                    height: ["20%", "100%", "20%"],
                  }}
                  transition={{
                    duration: 0.5 + Math.random() * 0.5,
                    repeat: Infinity,
                    delay: Math.random() * 0.5,
                  }}
                />
              ))}
            </div>
          )}
        </div>

        <div className="relative z-10 flex flex-col items-center gap-6">
          <div className={cn(
            "w-24 h-24 rounded-full flex items-center justify-center transition-all duration-500",
            recording && !paused ? "bg-red-50 text-red-500 shadow-[0_0_30px_rgba(239,68,68,0.2)]" : "bg-primary/5 text-primary"
          )}>
            {recording && !paused ? (
              <Mic className="w-10 h-10 animate-pulse" />
            ) : (
              <Mic className="w-10 h-10" />
            )}
          </div>

          <div className="text-4xl font-mono font-medium tabular-nums text-foreground/80">
            {formatTime(duration)}
          </div>

          {isProcessing && (
            <div className="flex items-center gap-2 text-sm text-accent">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Processing audio...</span>
            </div>
          )}

          {isFinalizing && (
            <div className="flex items-center gap-2 text-sm text-primary">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Finalizing transcription...</span>
            </div>
          )}

          <div className="flex items-center gap-4 mt-4">
            {!recording ? (
              <Button
                size="lg"
                className="h-14 px-8 rounded-full text-lg shadow-md hover:shadow-lg transition-all"
                onClick={handleStart}
                disabled={!isConnected && recording}
              >
                <Mic className="w-5 h-5 mr-2" />
                Start Recording
              </Button>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-14 w-14 rounded-full border-2"
                  onClick={paused ? resumeRecording : pauseRecording}
                >
                  {paused ? <Play className="w-6 h-6 fill-current" /> : <Pause className="w-6 h-6 fill-current" />}
                </Button>

                <Button
                  variant="destructive"
                  size="icon"
                  className="h-14 w-14 rounded-full shadow-md"
                  onClick={handleStop}
                  disabled={isFinalizing}
                >
                  <Square className="w-6 h-6 fill-current" />
                </Button>
              </>
            )}
          </div>
        </div>
      </Card>

      {/* Live Transcript Preview - Always show when recording */}
      <AnimatePresence>
        {recording && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between px-2">
              <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Live Transcription
              </span>
              {isProcessing && (
                <span className="flex items-center gap-1 text-xs text-accent font-medium">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Processing...
                </span>
              )}
            </div>
            <Card className="border-2 border-primary/10">
              <div className="p-6 min-h-[150px] max-h-[400px] overflow-y-auto">
                {transcript ? (
                  <div className="text-base leading-relaxed text-foreground whitespace-pre-wrap">
                    {transcript}
                    {!paused && <span className="inline-block w-2 h-5 bg-primary/50 ml-1 animate-pulse align-middle" />}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-[150px] text-muted-foreground">
                    <Loader2 className="w-8 h-8 animate-spin mb-3 text-primary/50" />
                    <p className="text-sm">Listening... Speak now to begin transcription</p>
                    {isConnected && (
                      <p className="text-xs mt-1 text-primary/70">Connected to transcription service</p>
                    )}
                  </div>
                )}
              </div>
            </Card>

            {/* Debug info */}
            {transcript && (
              <div className="text-xs text-muted-foreground text-right px-2">
                {transcript.split(' ').length} words • {transcript.length} characters
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

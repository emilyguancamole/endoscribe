import { useState, useRef, useCallback, useEffect } from 'react';
import { WEBSOCKET_FINALIZING_TIMEOUT_MS, PROCESSING_TRANSCRIPTION_TIMEOUT_MS } from '../constants';

export function useWebSocketTranscription(sessionId, setSessionId) {
  const [transcript, setTranscript] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const websocketRef = useRef(null);
  const closingTimerRef = useRef(null);
  const processingTimerRef = useRef(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/transcribe`;

    const ws = new WebSocket(wsUrl);
    websocketRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      ws.send(JSON.stringify({
        type: 'start',
        session_id: sessionId
      }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      console.log("WS: Message received")
      
      switch (message.type) {
        case 'status':
          if (message.session_id) {
            setSessionId(message.session_id);
          }
          if (message.message?.includes('Processing')) {
            setIsProcessing(true);
            if (processingTimerRef.current) {
              clearTimeout(processingTimerRef.current);
            }
            processingTimerRef.current = setTimeout(() => {
              setIsProcessing(false);
            }, PROCESSING_TRANSCRIPTION_TIMEOUT_MS);
          }
          break;

        case 'transcript':
          const text = message.data?.text || '';
          if (text) {
            setTranscript(prev => prev + ' ' + text);
            
            if (isFinalizing && closingTimerRef.current) {
              clearTimeout(closingTimerRef.current);
              closingTimerRef.current = setTimeout(() => {
                disconnect(false);
              }, WEBSOCKET_FINALIZING_TIMEOUT_MS);
            }

            setIsProcessing(true);
            if (processingTimerRef.current) {
              clearTimeout(processingTimerRef.current);
            }
            processingTimerRef.current = setTimeout(() => {
              setIsProcessing(false);
            }, PROCESSING_TRANSCRIPTION_TIMEOUT_MS);
          }
          break;

        case 'error':
          console.error('WebSocket error:', message.message);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed', event.code);
    };
  }, [sessionId, setSessionId, isFinalizing]);

  const disconnect = useCallback((graceful = true) => {
    if (graceful && websocketRef.current) {
      websocketRef.current.send(JSON.stringify({ type: 'end' }));
    } else {
      if (closingTimerRef.current) {
        clearTimeout(closingTimerRef.current);
        closingTimerRef.current = null;
      }
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
      setIsFinalizing(false);
    }
  }, []);

  const sendAudioChunk = useCallback((blob) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      console.log('Sending audio chunk:', blob.size, 'bytes');
      websocketRef.current.send(blob);
    } else {
      console.warn('WebSocket not ready, state:', websocketRef.current?.readyState);
    }
  }, []);

  const finalize = useCallback(() => {
    setIsFinalizing(true);
    disconnect(true);
    closingTimerRef.current = setTimeout(() => {
      disconnect(false);
    }, WEBSOCKET_FINALIZING_TIMEOUT_MS);
  }, [disconnect]);

  useEffect(() => {
    return () => {
      if (processingTimerRef.current) {
        clearTimeout(processingTimerRef.current);
      }
      if (closingTimerRef.current) {
        clearTimeout(closingTimerRef.current);
      }
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  return {
    transcript,
    isProcessing,
    isFinalizing,
    connect,
    disconnect,
    sendAudioChunk,
    finalize,
    clearTranscript: () => setTranscript('')
  };
}

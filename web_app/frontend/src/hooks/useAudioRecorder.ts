import { useState, useRef, useCallback } from 'react';
import RecordRTC from 'recordrtc';
import { AUDIO_CHUNK_INTERVAL_MS } from '../constants';

export function useAudioRecorder(onAudioChunk: (blob: Blob) => void) {
  const [recording, setRecording] = useState(false);
  const [paused, setPaused] = useState(false);
  const mediaRecorderRef = useRef<any>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new RecordRTC(stream, {
        type: 'audio',
        mimeType: 'audio/webm;codecs=opus',
        recorderType: RecordRTC.StereoAudioRecorder,
        timeSlice: AUDIO_CHUNK_INTERVAL_MS,
        ondataavailable: (blob: Blob) => {
          console.log('RecordRTC chunk received:', blob.size, 'bytes');
          if (blob.size > 0 && onAudioChunk) {
            onAudioChunk(blob);
          }
        }
      });

      recorder.startRecording();
      mediaRecorderRef.current = recorder;
      setRecording(true);
      setPaused(false);
      console.log('Recording started with chunk interval:', AUDIO_CHUNK_INTERVAL_MS, 'ms');
    } catch (error) {
      console.error('Error starting recording:', error);
      throw new Error('Failed to start recording. Please check microphone permissions.');
    }
  }, [onAudioChunk]);

  const pauseRecording = useCallback(() => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.pauseRecording();
      setPaused(true);
    }
  }, [recording]);

  const resumeRecording = useCallback(() => {
    if (mediaRecorderRef.current && paused) {
      mediaRecorderRef.current.resumeRecording();
      setPaused(false);
    }
  }, [paused]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stopRecording();
      setRecording(false);
      setPaused(false);
      mediaRecorderRef.current = null;

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  }, []);

  return {
    recording,
    paused,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording
  };
}

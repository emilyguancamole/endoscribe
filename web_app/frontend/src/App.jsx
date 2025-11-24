import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ProcedureSelector } from './components/ProcedureSelector';
import { AudioRecorder } from './components/AudioRecorder';
import { TranscriptionDisplay } from './components/TranscriptionDisplay';
import { ResultsDisplay } from './components/ResultsDisplay';
import { ErrorDisplay } from './components/ErrorDisplay';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import { useWebSocketTranscription } from './hooks/useWebSocketTranscription';
import { processTranscript as processTranscriptAPI, checkHealth } from './utils/api';
import { ERROR_DISPLAY_DURATION_MS } from './constants';

function App() {
  const [procedureType, setProcedureType] = useState('col');
  const [sessionId, setSessionId] = useState(null);
  const [healthStatus, setHealthStatus] = useState('checking');
  
  // UI states
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCalculatingPEP, setIsCalculatingPEP] = useState(false);
  const [recordingStatus, setRecordingStatus] = useState('');
  
  // Results states
  const [showResults, setShowResults] = useState(false);
  const [showPEPRiskResults, setShowPEPRiskResults] = useState(false);
  const [colonoscopyData, setColonoscopyData] = useState({});
  const [polypsData, setPolypsData] = useState([]);
  const [procedureData, setProcedureData] = useState({});
  const [pepRiskData, setPepRiskData] = useState({});

  // Custom hooks
  const websocket = useWebSocketTranscription(sessionId, setSessionId);
  
  const handleAudioChunk = useCallback((blob) => {
    websocket.sendAudioChunk(blob);
  }, [websocket]);
  
  const audioRecorder = useAudioRecorder(handleAudioChunk);

  useEffect(() => {
    checkHealth()
      .then(data => {
        setHealthStatus(data.status === 'healthy' ? 'healthy' : 'degraded');
      })
      .catch(() => {
        setHealthStatus('error');
      });
  }, []);

  const showError = useCallback((message) => {
    setError(message);
    setTimeout(() => setError(''), ERROR_DISPLAY_DURATION_MS);
  }, []);

  // Recording handlers
  const handleStartRecording = async () => {
    try {
      websocket.clearTranscript();
      setShowResults(false);
      setShowPEPRiskResults(false);
      websocket.connect();
      await audioRecorder.startRecording();
      setRecordingStatus('Recording');
    } catch (err) {
      showError(err.message);
    }
  };

  const handlePauseRecording = () => {
    audioRecorder.pauseRecording();
    setRecordingStatus('Paused');
  };

  const handleResumeRecording = () => {
    audioRecorder.resumeRecording();
    setRecordingStatus('Recording');
  };

  const handleStopRecording = () => {
    audioRecorder.stopRecording();
    websocket.finalize();
    setRecordingStatus('Finalizing...');
    setTimeout(() => setRecordingStatus(''), 2000);
  };

  // Processing handlers
  const handleSubmit = async () => {
    const transcript = websocket.transcript.trim();
    if (!transcript) {
      showError('No transcript available to process');
      return;
    }

    setIsSubmitting(true);
    setShowPEPRiskResults(false);

    try {
      const result = await processTranscriptAPI(transcript, procedureType, sessionId);
      
      if (result.success) {
        if (procedureType === 'col') {
          setColonoscopyData(result.data.colonoscopy || {});
          setPolypsData(result.data.polyps || []);
        } else {
          setProcedureData(result.data);
        }
        setShowResults(true);
      } else {
        showError(result.error || 'Processing failed');
      }
    } catch (err) {
      showError('Failed to process transcript: ' + err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCalculatePEPRisk = async () => {
    const transcript = websocket.transcript.trim();
    if (!transcript) {
      showError('No transcript available to process');
      return;
    }

    setIsCalculatingPEP(true);
    setShowPEPRiskResults(false);

    try {
      const result = await processTranscriptAPI(transcript, 'pep_risk', sessionId);
      
      if (result.success) {
        const filteredData = {};
        for (const [key, value] of Object.entries(result.data)) {
          if (key !== 'id' && key !== 'model') {
            filteredData[key] = value;
          }
        }
        setPepRiskData(filteredData);
        setShowPEPRiskResults(true);
      } else {
        showError(result.error || 'PEP Risk calculation failed');
      }
    } catch (err) {
      showError('Failed to calculate PEP risk: ' + err.message);
    } finally {
      setIsCalculatingPEP(false);
    }
  };

  const showSubmitButton = !audioRecorder.recording && !websocket.isFinalizing && websocket.transcript.trim().length > 0;
  const showPEPRiskButton = procedureType === 'ercp' && showSubmitButton;

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <Header healthStatus={healthStatus} />
      
      <ProcedureSelector 
        value={procedureType} 
        onChange={setProcedureType} 
      />

      <AudioRecorder
        recording={audioRecorder.recording}
        paused={audioRecorder.paused}
        onStart={handleStartRecording}
        onPause={handlePauseRecording}
        onResume={handleResumeRecording}
        onStop={handleStopRecording}
        sessionId={sessionId}
        recordingStatus={recordingStatus}
      />

      <TranscriptionDisplay
        transcript={websocket.transcript}
        isProcessing={websocket.isProcessing}
        showSubmitButton={showSubmitButton}
        showPEPRiskButton={showPEPRiskButton}
        onSubmit={handleSubmit}
        onCalculatePEPRisk={handleCalculatePEPRisk}
        isSubmitting={isSubmitting}
        isCalculatingPEP={isCalculatingPEP}
      />

      <ResultsDisplay
        procedureType={procedureType}
        colonoscopyData={colonoscopyData}
        polypsData={polypsData}
        procedureData={procedureData}
        pepRiskData={pepRiskData}
        showResults={showResults}
        showPEPRiskResults={showPEPRiskResults}
      />

      <ErrorDisplay message={error} />
    </div>
  );
}

export default App;

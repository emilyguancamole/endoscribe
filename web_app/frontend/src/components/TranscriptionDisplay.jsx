import { useEffect, useRef } from 'react';

export function TranscriptionDisplay({ 
  transcript, 
  isProcessing,
  showSubmitButton,
  showPEPRiskButton,
  onSubmit,
  onCalculatePEPRisk,
  isSubmitting,
  isCalculatingPEP
}) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [transcript]);

  const displayText = transcript.trim() || 'Transcription will appear here as you speak...';

  return (
    <div className="card bg-base-100 shadow-xl mb-4">
      <div className="card-body">
        <h2 className="card-title flex justify-between items-center w-full">
          <span>Transcription</span>
          {isProcessing && (
            <span className="loading loading-dots loading-sm"></span>
          )}
        </h2>

        <div 
          ref={containerRef}
          className="bg-base-200 p-4 rounded-lg min-h-[200px] max-h-[400px] overflow-y-auto"
        >
          <p className="whitespace-pre-wrap break-words text-base-content/70">
            {displayText}
          </p>
        </div>

        {showSubmitButton && (
          <div className="flex gap-4 mt-4">
            <button 
              onClick={onSubmit}
              disabled={isSubmitting}
              className="btn btn-primary"
            >
              {isSubmitting && <span className="loading loading-spinner"></span>}
              <span>{isSubmitting ? 'Processing...' : 'Submit for Processing'}</span>
            </button>

            {showPEPRiskButton && (
              <button 
                onClick={onCalculatePEPRisk}
                disabled={isCalculatingPEP}
                className="btn btn-secondary"
              >
                {isCalculatingPEP && <span className="loading loading-spinner"></span>}
                <span>{isCalculatingPEP ? 'Calculating...' : 'Calculate PEP Risk'}</span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

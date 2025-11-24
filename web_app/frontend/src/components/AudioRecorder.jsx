export function AudioRecorder({ 
  recording, 
  paused, 
  onStart, 
  onPause, 
  onResume, 
  onStop,
  sessionId,
  recordingStatus 
}) {
  const statusClass = recording && !paused ? 'badge-error' : 
                      paused ? 'badge-warning' : 'badge-neutral';

  return (
    <div className="card bg-base-100 shadow-xl mb-4">
      <div className="card-body">
        <h2 className="card-title flex justify-between items-center w-full">
          <span>Audio Recording</span>
          {recordingStatus && (
            <span className={`badge ${statusClass}`}>{recordingStatus}</span>
          )}
        </h2>

        <div className="flex gap-4 items-center">
          {!recording && (
            <button onClick={onStart} className="btn btn-primary">
              Start Recording
            </button>
          )}

          {recording && !paused && (
            <button onClick={onPause} className="btn btn-warning">
              Pause
            </button>
          )}

          {recording && paused && (
            <>
              <button onClick={onResume} className="btn btn-success">
                Resume
              </button>
              <button onClick={onStop} className="btn btn-error">
                Finish
              </button>
            </>
          )}
        </div>

        {sessionId && (
          <div className="text-sm text-base-content/70">
            Session ID: <span className="font-mono">{sessionId}</span>
          </div>
        )}
      </div>
    </div>
  );
}

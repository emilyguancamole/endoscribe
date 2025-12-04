import { Mic, PauseCircle, PlayCircle, Square } from "lucide-react";

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

  const statusClass = recording && !paused
    ? "badge-error"
    : paused
      ? "badge-warning"
      : "badge-neutral";

  return (
    <div className="card bg-base-100 shadow-sm border border-base-200">
      <div className="card-body space-y-1">

        <div className="flex justify-between items-center">
          <h2 className="card-title text-lg font-semibold flex items-center">
            Audio Recording
          </h2>
          {recordingStatus && (
            <span className={`badge ${statusClass}`}>{recordingStatus}</span>
          )}
        </div>

        {!recording && (
          <button
            onClick={onStart}
            className="btn btn-primary btn-lg w-full"
          >
            <Mic className="w-5 h-5" />
            Start Recording
          </button>
        )}

        {recording && !paused && (
          <div className="flex gap-3">
            <button onClick={onPause} className="btn btn-warning flex-1">
              <PauseCircle className="w-5 h-5" />
              Pause
            </button>
            <button onClick={onStop} className="btn btn-error flex-1">
              <Square className="w-5 h-5" />
              Finish
            </button>
          </div>
        )}
        {recording && paused && (
          <div className="flex gap-3">
            <button onClick={onResume} className="btn btn-success flex-1">
              <PlayCircle className="w-5 h-5" />
              Resume
            </button>
            <button onClick={onStop} className="btn btn-error flex-1">
              <Square className="w-5 h-5" />
              Finish
            </button>
          </div>
        )}

        {sessionId && (
          <div className="text-xs text-base-content/60 mt-1">
            Session ID: <span className="font-mono">{sessionId}</span>
          </div>
        )}
      </div>
    </div>
  );
}

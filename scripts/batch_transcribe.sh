#!/usr/bin/env bash
# 12/13/2025 - created to transcribe a bunch of files in IA1 with whisperx
# Batch run transcription.transcription_service sequentially over audio files.
# Usage examples:
#  ./scripts/batch_transcribe.sh --input-dir transcription/recordings/ercp/bdstone --procedure-type ercp --service whisperx
#  ./scripts/batch_transcribe.sh --input-dir recordings/ercp --procedure-type ercp --service whisperx --ext wav --log-dir logs/transcription --dry-run

set -euo pipefail

INPUT_DIR=""
PROCEDURE_TYPE="ercp"
SERVICE="whisperx"
EXT="wav"
LOG_DIR="logs/transcription"
DRY_RUN=false
SKIP_IF_LOG_EXISTS=true
SLEEP_SEC=0

usage() {
  cat <<EOF
Usage: $0 --input-dir <dir> [--procedure-type <proc>] [--service <svc>] [--ext <wav>] [--log-dir <dir>] [--dry-run] [--no-skip]

Options:
  --input-dir       Directory containing audio files to process (required)
  --procedure-type  Procedure type to pass to transcription service (default: ercp)
  --service         Transcription service to use (default: whisperx)
  --ext             File extension to process (default: wav)
  --log-dir         Directory to write per-file logs (default: logs/transcription)
  --dry-run         Print commands but don't execute
  --no-skip         Do not skip files with existing logs (by default, existing logs are skipped)
  --sleep-seconds   Seconds to sleep between runs (default: 0)
  -h, --help        Show this help

Example:
  $0 --input-dir transcription/recordings/ercp/bdstone --procedure-type ercp --service whisperx
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"; shift 2;;
    --procedure-type)
      PROCEDURE_TYPE="$2"; shift 2;;
    --service)
      SERVICE="$2"; shift 2;;
    --ext)
      EXT="$2"; shift 2;;
    --log-dir)
      LOG_DIR="$2"; shift 2;;
    --dry-run)
      DRY_RUN=true; shift 1;;
    --no-skip)
      SKIP_IF_LOG_EXISTS=false; shift 1;;
    --sleep-seconds)
      SLEEP_SEC="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  echo "--input-dir is required" >&2
  usage
  exit 2
fi

mkdir -p "$LOG_DIR"

# Find files (non-recursive by default). Use globbing to support spaces.
shopt -s nullglob
FILES=("$INPUT_DIR"/*."$EXT")
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No *.$EXT files found in $INPUT_DIR"
  exit 0
fi

echo "Found ${#FILES[@]} .$EXT files in $INPUT_DIR. Logs -> $LOG_DIR"

for file in "${FILES[@]}"; do
  # derive a sample name
  base=$(basename "$file")
  sample="${base%.*}"
  logfile="$LOG_DIR/${sample}.log"

  if [[ "$SKIP_IF_LOG_EXISTS" = true && -f "$logfile" ]]; then
    echo "Skipping $file because $logfile exists"
    continue
  fi

  cmd=(python -m transcription.transcription_service --procedure_type "$PROCEDURE_TYPE" --audio_file "$file" --service "$SERVICE")

  echo "Running: ${cmd[*]}"
  if [[ "$DRY_RUN" = true ]]; then
    echo "DRY RUN: would run: ${cmd[*]}"
  else
    # Run and capture output
    if "${cmd[@]}" 2>&1 | tee "$logfile"; then
      echo "Completed $file -> $logfile"
    else
      echo "Command failed for $file. See $logfile for details" >&2
      # keep going to next file (don't exit on first failure)
    fi
  fi

  if [[ "$SLEEP_SEC" -gt 0 ]]; then
    sleep "$SLEEP_SEC"
  fi
done

echo "Batch transcription finished. Logs are under $LOG_DIR"

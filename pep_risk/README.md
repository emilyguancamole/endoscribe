# PEP Risk Pipeline (pep_risk)

AI-powered extraction of PEP risk factors from a procedure dictation. This is a related, but largely separate project from the main EndoScribe project.

## Features
- **Live audio recording**: Record and upload audio in real time, chunked for streaming transcription.
- **Pre-recorded file support**: Rather than real-time transcription, can also process full-length wav/m4a/mp3 files for extraction. Used for testing.
- **LLM-based extraction**: Uses LLM to extract structured ERCP fields from transcripts.
- **Ground truth evaluation**: Compares LLM extraction to ground truth CSV for accuracy metrics.
- **Results saved**: All outputs and evaluations are saved as CSVs in `pep_risk/results_longform`.

## How to Run
### Prereqs
WhisperX transcription needs GPUs. I run the PEP server on a remote server with GPUs.

### 1. Start the server
(After sshing into GPU server):
```bash
uvicorn pep_risk.server:app --host 0.0.0.0 --port 8000
```

### 2. Live recording (chunked upload)
```bash
python pep_risk/record_and_upload.py
```
- Press Ctrl+C to finalize and trigger LLM extraction.

### 3. Process a pre-recorded file (server-side)
Place your file in `pep_risk/recordings/` and run:
```bash
curl -F "filename=yourfile.mp3" http://localhost:8000/process_local
```

## Outputs
- Extracted results: `pep_risk/results_longform/pep_eval.csv`
- Evaluation metrics: `pep_risk/results_longform/ercp_eval.csv`

## Design Choices
- **Chunked live upload**: By sending recordings in chunks, this reduces latency for transcribing long procedures and avoids memory issues.
    - TODO: in real procedures, increase the chunk length. Currently set to 30sec for testing, which is too short.
- **Text-based finalization**: For live transcription, only the final transcript (concatenated from all chunks) is sent to the LLM, minimizing API calls and cost.
- **Simple evaluation**: Ground truth comparison by comparing to corresponding fields in RedCap (exported CSVs).

## Requirements
- Python 3.10+
- FastAPI, Uvicorn, WhisperX, torch, pandas, requests
- GPUs for faster WhisperX inference
- Azure/OpenAI API key for LLM extraction

## Notes
- Update `map_ground_truth_columns` in `server.py` for different risk factor/column names.

## TODO
As of 10/19/2025:
- Incorporate non-LLM risk factors from RedCap/EHR into results.
- PEP risk model needs to be implemented / incorporated: take in risk factors and output PEP risk.

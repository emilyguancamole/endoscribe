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
(After sshing into GPU server, IA1):
```bash
uvicorn pep_risk.server:app --host 0.0.0.0 --port 8001
```

### 2.1 Live recording (chunked upload)
```bash
python pep_risk/record_and_upload.py
```
- Press Ctrl+C to finalize and trigger LLM extraction.

### 2.2 Pre-recorded file (server-side)
Place mp3/m4a/wav file in `pep_risk/recordings/` and run (for example):
```bash
curl -F "filename=11864-10.m4a" http://localhost:8001/process_local
```
Response includes: session_id, transcript, extraction dict, finalized=true, evaluation: overall_accuracy + per_field results, matched by column record_id.
?If you want to process multiple files under one session, you can pass session_id in the form and call /process_local repeatedly; it will concatenate the transcripts across those calls before extraction. If you want each file independent, omit session_id each time.

### 3. Evaluation against ground truth
Evaluation is done using functions in `pep_risk/evaluation.py`. It's done automatically as part of the server. But if error, eval can be run separately using saved extraction csvs (via `pep_risk/evaluate_only.py`).

Evaluate using saved extraction from sessions CSV
`python pep_risk/evaluate_only.py 1234`

Provide extraction JSON directly
`python pep_risk/evaluate_only.py 1234 --extraction-json path/to/extraction.json`

Save results to JSON
`python pep_risk/evaluate_only.py 1234 --out-json results/eval.json`

Custom paths
python pep_risk/evaluate_only.py 1234 \
  --sessions-csv my_results/sessions.csv \
  --ground-truth my_data/ground_truth.csv


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

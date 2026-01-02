# EndoScribe
## Introduction

EndoScribe is an AI-powered scribe for automating endoscopy documentation.

Currently, endoscopists' note-writing workflow looks like this: 
> Perform the procedure and remember procedure details -> Write a note by heavily editing a template (EndoPro, Provation), filling it in with procedure details from memory.

This workflow requires physicians to spend signficant time on documentation.

EndoScribe automates note-writing. With EndoScribe's help, the workflow now looks like this: 
> Perform and dictate the procedure -> Review + edit an AI-generated procedure.

EndoScribe gives time back to the physician, reducing burnout and allowing them to spend more time with patients.

This README outlines how the code is organized, rationale/details of implementation, and how to run the main stages of the scribe.

## High-level architecture

An AI-powered scribe for automating endoscopy documentation. At a high level, this scribe works in 3 parts:
1. **Transcription**
    - Audio dictations of the procedure are input to **Transcription** module. This optionally pre-processes the audio input, then uses Whisper to transcribe into text word-for-word. It outputs text transcripts as CSV files.
2. **Processors**
    - The transcripts are fed into **Processors**. Processors build a prompt, prompt an LLM to extract data from the transcripts, and perform validation of the extracted data. Data is saved as CSV and optionally written to Postgres.
3. **Drafters**
    - The data is fed into **Drafters**, which formats the final note. This includes formatting data extracted directly by the LLM, creating follow-up recommendations based on certain findings in the data, and integrating patient information. The drafter outputs a final note draft as a `.docx`.

The scribe currently processes 4 types of endoscopy procedures: Colonoscopy, EGD, ERCP, and EUS. Logic across these procedure types are similar, with key differences. Therefore, several folders in this repo (i.e. `processors`, `prompts`, `results`, `drafters`) are organized by procedure type.

## Repository Structure
# EndoScribe

## Introduction

EndoScribe is an AI-powered scribe for automating endoscopy documentation. It converts recorded procedure dictations into structured data and draft clinical notes to reduce physician documentation time.

This README documents the current (Dec 2025) state of the project and how to run the three main stages:
- Transcription (audio -> transcripts)
- LLM Extraction (transcripts -> structured data)
- Drafting/Templating (structured data -> `.docx` draft)


## High-level architecture

0.5. Template generation
1. Transcription
   - Converts raw audio into textual transcripts.
   - Backends supported:
     - WhisperX (local GPU-based, offline, highly customizable)
     - Azure Speech Service (cloud-managed, scalable, supports phrase lists)
   - A unified interface (`transcription/transcription_service.py`) selects the backend and forwards options such as `procedure_type`, `phrase_list`, and `save_filename`.
   - Transcripts are saved under `transcription/results/{procedure_type}/` or a specified filename.

2. Processors / LLM Extraction
   - Processors (in `processors/`) build prompts from `prompts/{procedure}/`, call an LLM via `llm/llm_client.py`, parse the result, and validate with Pydantic models (`models/`).
   - Default local LLM: Llama via `vllm`; adapters allow other providers.

3. Drafters / Templating
   - Drafters (in `drafters/`) format extracted data into a clinical note draft (`.docx`).
   - Output saved in `drafters/results/{procedure}/`.

3.5 Reviewer (planned)
   - A future module to review and edit drafted notes.

## Transcription

Supported entry points
- `transcription/transcription_service.py` — unified API, auto-selects Azure or WhisperX --> NOT YET VALIDATED 12/7/25
- `transcription/azure_transcribe.py` — Azure Speech Service implementation
- `transcription/whisperx_transcribe.py` — WhisperX implementation

Storage and filenames
- Default aggregated CSV for a procedure: `transcription/results/{procedure_type}/transcriptions.csv`
- If you pass `save_filename`, transcription will write to `transcription/results/{procedure_type}/{save_filename}.csv` (or `_misc` if no `procedure_type`).

Phrase lists and medical vocabulary
- Azure supports `PhraseListGrammar` to bias recognition towards medical terms. Place one phrase per line in:
  - `prompts/{procedure_type}/phrases.txt` or
  - `prompts/{procedure_type}_phrases.txt`
- If no phrase file exists the system uses a small built-in fallback list for common procedures (`ercp`, `col`, `egd`, `eus`).

Examples
- Run Azure transcription and save to a named file:
```bash
python -m transcription.azure_transcribe --audio_file path/to/audio.wav --procedure_type ercp --save_filename=run-2025-12-07
```
- Unified programmatic API:
```python
from transcription.transcription_service import transcribe_unified
res = transcribe_unified(
    "audio.wav",
    service="azure",
    procedure_type="ercp",
    save_filename="run-2025-12-07"
)
```


## LLM Extraction (processors)

Key files
- `processors/` — per-procedure processors (Col, ERCP, EGD, EUS)
- `llm/llm_client.py` — LLM client adapter (local or remote)
- `prompts/{procedure}/` — `system.txt`, `field_definitions.txt`, `fewshot/`
- `models/` — Pydantic models for validation

Workflow
1. Processor builds messages from prompt files and transcript text.
2. Calls `LLMClient` for a completion.
3. Parses and validates LLM output into structured records.
4. Writes validated outputs to `results/{procedure}/` and optionally to Postgres.

Run an extraction example
```bash
python main.py --procedure_type=col --transcripts_fp=transcription/results/col/run-2025-12-07.csv --output_filename=run-2025-12-07 --files_to_process all
```

Notes
- Prompts and fewshot examples control extraction behavior; adjust them to influence the LLM's output format.


## Drafting / Templating

Files
- `drafter.py`

What it does
- Converts structured LLM-extracted data into a formatted clinical note (`.docx`).

Example
```bash
python drafter.py --procedure=col --pred_csv=results/col/run-2025-12-07_colonoscopies.csv --polyp_csv=results/col/run-2025-12-07_polyps.csv --output_dir=drafters/results/col --samples_to_process all
```


## Environment & dependencies

Python 3.10+ recommended.

Install dependencies:
```bash
pip install -r requirements.txt
# Using vllm/WhisperX needs GPU drivers and suitable torch/vllm installs.
```

Have an `.env` file; `.env.example` included as a template.

## Migration notes

- The repo is currently undergoing a migration from WhisperX (on-prem) to Azure (managed).


## Future steps

- Add automated Azure vs WhisperX comparison scripts.
- Replace aggregated CSVs with a database and background workers for production.



## Contact

Repo owner: Emily Guan — emilymguan@gmail.com

README last updated: 12/7/2025
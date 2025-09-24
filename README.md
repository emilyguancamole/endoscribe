# EndoScribe
## Introduction

EndoScribe is an AI-powered scribe for automating endoscopy documentation.

Currently, endoscopists' note-writing workflow looks something like this: perform the procedure and remember procedure details -> write a note by heavily editing a template, filling it in with procedure details.

This workflow requires physicians to spend signficant time on documentation.

EndoScribe automates note-writing. The workflow now looks like this: perform and dictate the procedure -> review an AI-generated note draft of the procedure.

EndoScribe's goal is to give time back to the physician, reducing burnout and allowing them to spend more time with patients.

This README outlines how the code is organized, rationale/details of implementation, and how to run the three main stages of the scribe (transcription -> extraction/processors -> drafting).

## High-level architecture

An AI-powered scribe for automating endoscopy documentation. At a high level, this scribe works in 3 parts:
1. **Transcription**
    - Audio dictations of the procedure are input to **Transcription** module. This optionally pre-processes the audio input, then uses Whisper to transcribe into text word-for-word. It outputs text transcripts as CSV files.
2. **Processors**
    - The transcripts are fed into **Processors**. Processors build a prompt, prompt an LLM to extract data from the transcripts, and perform validation of the extracted data. Data is saved as CSV and optionally written to Postgres.
3. **Drafters**
    - The data is fed into **Drafters**, which formats the final note. This includes formatting data extracted directly by the LLM, as well as creating follow-up recommendations based on certain findings in the data. The drafter outputs a final note draft as a `.docx`.

The scribe currently processes 4 types of endoscopy procedures: Colonoscopy, EGD, ERCP, and EUS. Logic across these procedure types are similar, with key differences. Therefore, several folders in this repo (i.e. `processors`, `prompts`, `results`, `drafters`) are organized by procedure type.

## Repository Structure
Below is a simplified file tree for this repo. 

```
.
├── README.md
├── main.py                 # main entry point for Part 2 (processors)
├── data_models             # Pydantic models for each procedure's extracted data
├── db                      # Postgres writer/upsert helpers
├── drafter.py              # Main entry point for Part 3 (drafters)
├── drafters                # Converts model data into Word docs
├── llm                     # LLM client (vllm/llama) and prompt helpers
├── processors              # Build prompts, call LLM, validate, save outputs
├── prompts                 # System instructions, data field definitions, fewshot examples
├── results                 # CSV outputs produced by processors
└── transcription
    ├── results                 # Text transcripts as CSV
    └── whisper_transcribe.py   # Main entry + logic for Part 1 (transcription)
```


## How to Run in Development
NOTE: A case example is found in `demo.ipynb`, using one procedure to run the full pipeline in a Jupyter notebook.

Prerequisites
- Python 3.10+ (project uses 3.10 in development)
- GPU + CUDA for local Llama/Whisper acceleration

Install dependencies. Note that `requirements.txt` includes heavy packages like `vllm` and `torch`:

```bash
python -m pip install -r requirements.txt
```

Run transcription on a folder of audio files:
```bash
python -m transcription.whisper_transcribe --procedure_type=col --save_filename=demo_whisper_lg --model=openai/whisper-large-v3 --audio_dir=/path/to/audio_files_folder
```

This generates a CSV under `transcription/results/{procedure_type}/{save_filename}.csv` with the transcripts.

Run extraction for a procedure type (example: colonoscopy):
```bash
python main.py --procedure_type=col --transcripts_fp=dev_transcripts_file.csv --output_filename=demo_llm_output --files_to_process all
```

- `main.py` wires up `LLMHandler` (vllm/llama), loads prompt files from `prompts/col/`, and writes CSV outputs to `results/col/`. Note that for colonoscopies, 2 output CSVs will be generated, with `colonoscopies` and `polyps` appended to `output_filename`. For all other procedures, 1 output CSV will be generated with name `{output_filename}.csv`.

Draft a Word document from extracted CSVs (example: colonoscopy):
```bash
python drafter.py --procedure=col --pred_csv=results/col/demo_llm_output_colonoscopies.csv --polyp_csv=results/col/demo_llm_output_polyps.csv --output_dir=drafters/results/col --samples_to_process all
```


## Implementation Details

### Part 1: Transcription
Purpose: convert raw audio recordings into reliable, time-ordered textual transcripts.

Approach: 
- Lightweight preprocessing (stereo->mono conversion and resampling). I experimented with other utils further preprocessing (e.g. vocal separation), but currently am not using.
- Whisper model for transcription (usually use whisper-lg-v3) pulled from HuggingFace.
- Outputs CSV rows with at least these columns: `file` (string id), `pred_transcript` (transcribed text).

Future transcription-related work I'm exploring:
- Heavier preprocessing for full-procedure-length audio, such as removing silent segments before transcription.
- Noise suppression.
- Using **WhisperX**: built on top of faster-whisper and includes preprocessing, e.g. voice activity detection and speaker diarization.


### Part 2: Processors

**Purpose:** Data extraction from text transcripts. 
Each procedure type (col, egd, ercp, eus) has its own Processor subclass that implements parsing/shape-specific logic.

**LLM**
- Default LLM: local Llama via `vllm` (see `llm/llm_handler.py`). This gives low-latency, on-prem inference for sensitive clinical data. An Azure OpenAI or other backend can be added with a thin adapter.
- Processors build a list of messages in chat format and pass them to the `llm_handler.chat(messages)` method. LLM is expected to return a textual completion that can be post-processed into JSON.

**Prompts**
- Files live in `prompts/{procedure}/` and are composed of:
    - a small field-definition file (`colonoscopies.txt`, `eus.txt`, etc.) describing expected extraction fields and their formats;
    - `system.txt` which becomes the system message and includes field definitions;
    - `fewshot/` which contains `*_user.txt` / `*_assistant.txt` pairs used as conversational few-shot examples.

**Steps inside a Processor** (e.g. `ColProcessor`):
    1. Call `build_messages(...)` (BaseProcessor) to create system + fewshot + user messages.
    2. Send messages to `llm_handler.chat(...)` and retrieve the text completion.
    3. Parse JSON and validate with the appropriate Pydantic model in `data_models/data_models.py`.
    4. Convert/clean types, append to outputs list, and finally call `save_outputs(...)`.

**Error modes and handling**
- JSON parsing failures: we currently skip the sample and continue. 
- Validation failures: Pydantic `ValidationError` is logged; by default the sample is skipped.
- Future: add retries or a secondary cleanup LLM call.

**Important note about Colonoscopy procedures:**
- Colonoscopy procedures have 2 processing parts.
    - `colonoscopies` extracts core procedure details. 
    - `polyps` extracts polyp-level output with sizes, locations, classification of polyps found during the procedure. 
    - These are separated to allow futher analysis with just the polyp findings. This means 2 LLM calls are made for each Colonoscopy procedure extraction.
- All other procedures have just 1 processing part.
    - For example, each EUS procedure has 1 call to the LLM to extract EUS procedure details.


### Part 3: Drafters

**Purpose:** Turn structured data into a readable clinical draft (.docx). Drafters don't need to understand LLM or extraction internals—they only receive data records.

**Key notes**
- Each drafter implements `EndoscopyDrafter` in `drafters/base.py`.
- Drafters expect the `pred_df` index to be the sample `id`. They use `self._get_sample_row(...)` to pick a single sample row and then render sections (Indications, Findings, Impressions, Recommendations, Repeat Exam).
- Formatting: we use `python-docx` to add headings, paragraphs, and more complex inline formatting (for example `drafters/utils.py` contains helper logic to bold subheadings inside long `findings` text).

**Recall / Recommendations**
- Beyond formatting data into drafts, drafters also contain small, deterministic decision rules to construct recall and/or recommendations sections from extracted data.
    - Example: suggests colonoscopy recall intervals based on `polyp_count` and `size_max_mm`.


## Anticipated Changes Needed
From dev to production, we'll need some changes. These may include:
- Heavier audio pre-processing and changes to transcription pipeline, as described above.
- Replace local CSV wiring with tighter database use and message queue / API for near-real-time processing. End goal is to have near-real-time transcription; note generation done post-procedure.
- Automated validation of LLM-extracted data. Currently, validation of generated notes is done with human reviewers.
- Major future tasks: integration with EHR.

## Contact + Notes
Feel free to contact the owner of the repo, Emily Guan, with any questions. Email: emilymguan@gmail.com



**Note:**

Some old files (e.g. GPT extraction Python files) are only in the [old repository](https://github.com/emilyguancamole/endoscribe-old). I don't anticipate ever needing them again, but they exist as history. (Old repo also has a more complete commit history.)

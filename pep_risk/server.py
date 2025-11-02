import traceback
from fastapi import FastAPI, UploadFile, Form
import os, uuid, asyncio
from whisperx import load_model
import torch
from typing import Dict, List, Optional
from transcription.whisperx_transcribe import transcribe_whisperx
from llm.llm_client import LLMClient
from processors.ercp_processor import ERCPProcessor
import pandas as pd
from datetime import datetime
from pep_risk.evaluation import evaluate_against_ground_truth
from fastapi.encoders import jsonable_encoder

app = FastAPI()

UPLOAD_DIR = "uploads"
RESULTS_DIR = "pep_risk/results_longform"
RECORDINGS_DIR = "pep_risk/recordings"
GROUND_TRUTH_CSV = "pep_risk/ground_truth.csv"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Load WhisperX once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("large-v3", device=device)

# Simple in-memory session store for chunked recording sessions
# session_id -> {"chunks": [file_paths], "transcripts": [texts], "finalized": bool, "final_transcript": Optional[str], "extraction": Optional[dict]}
SESSIONS: Dict[str, Dict[str, Optional[object]]] = {}

# Initialize LLM + ERCP processor once
try:
    llm_handler = LLMClient.from_config("openai_gpt4o")
except Exception as e:
    # Fallback: allow env to override or raise
    print("Failed to initialize LLM from config 'openai_gpt4o':", e)
    llm_handler = None

ercp_processor: Optional[ERCPProcessor] = None
if llm_handler is not None:
    try:
        ercp_processor = ERCPProcessor(
            procedure_type="ercp",
            system_prompt_fp="pep_risk/prompts/system.txt",
            output_fp="pep_risk/results/pep_server.csv",
            llm_handler=llm_handler,
            to_postgres=False,
        )
    except Exception as e:
        print("Failed to initialize ERCPProcessor:", e)
        ercp_processor = None


async def process_pipeline(transcript: str) -> dict:
    """Run ERCP extraction via LLM and return the validated dict of fields."""
    if ercp_processor is None:
        raise RuntimeError("ERCP processor is not initialized; check LLM config and credentials.")
    # Run once on the final transcript
    # Use the implemented single-transcript extractor
    result = ercp_processor.extract_pep_from_transcript(transcript, filename="session")
    return result


def simple_pep_rule(transcript: str, extraction: Optional[dict]) -> str:
    """A placeholder risk label derived from transcript/extraction; replace with a real model later."""
    text = (transcript or "").lower()
    if extraction and isinstance(extraction, dict):
        # crude signals from extraction
        pd_stent = extraction.get("pd_stent")
        ercp_findings = str(extraction.get("ercp_findings", "")).lower()
        if "pancreatic duct" in ercp_findings and not pd_stent:
            return "High"
    if "cannulation" in text:
        return "Moderate"
    return "Low"


def save_session_results(session_id: str, final_transcript: str, extraction: dict):
    """Save the session results (transcript + ERCP extraction) to a CSV file."""
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Prepare the data row
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_row = {
        "session_id": session_id,
        "timestamp": timestamp,
        "transcript": final_transcript,
        "transcript_length": len(final_transcript),
        **extraction  # Unpack all ERCP extraction fields
    }
    
    # Define the output CSV file path
    csv_path = os.path.join(RESULTS_DIR, "pep_llm_extraction.csv")
    
    # Load existing data if file exists, otherwise create new DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved PEP risk extraction results to {csv_path}")


async def finalize_session(session_id: str):
    """Finalize a recording session: concatenate all chunk transcripts and run LLM extraction."""
    if session_id not in SESSIONS:
        raise ValueError(f"Unknown session_id: {session_id}")
    session = SESSIONS[session_id]
    if session.get("finalized") and session.get("final_transcript"):
        return session["final_transcript"], session.get("extraction")

    transcripts = session.get("transcripts", [])
    if not transcripts:
        raise ValueError("No transcripts available for this session.")

    # Concatenate all chunk transcripts with space separator
    print(f"Concatenating {len(transcripts)} transcript chunks...")
    final_text = " ".join(transcripts).strip()
    
    print(f"Final transcript length: {len(final_text)} characters")
    print("Running LLM extraction...")
    extraction = await process_pipeline(final_text)

    session["final_transcript"] = final_text
    session["extraction"] = extraction
    session["finalized"] = True
    
    # Save results to CSV
    save_session_results(session_id, final_text, extraction)
    
    return final_text, extraction


@app.post("/upload")
async def upload_audio(
    file: UploadFile,
    session_id: Optional[str] = Form(None),
    is_last: Optional[str] = Form("false"),
):
    """Receive audio chunk, attach to a recording session, and transcribe the chunk.
    If is_last is true, finalize: merge audio, produce final transcript once, and run ERCP extraction.
    """
    # Save file (keep original extension)
    file_id = str(uuid.uuid4())
    orig_name = getattr(file, "filename", None) or ""
    _, ext = os.path.splitext(orig_name)
    ext = (ext or ".wav").lower()
    allowed_exts = {".wav", ".m4a", ".mp3"}
    if ext not in allowed_exts:
        ext = ".wav"  # default if unknown
    audio_fp = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(audio_fp, "wb") as f:
        f.write(await file.read())

    # Manage session
    if not session_id:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"chunks": [], "transcripts": [], "finalized": False, "final_transcript": None, "extraction": None}
    else:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = {"chunks": [], "transcripts": [], "finalized": False, "final_transcript": None, "extraction": None}
    SESSIONS[session_id]["chunks"].append(audio_fp)

    # Transcribe current chunk for quick feedback
    print(f"Transcribing chunk {audio_fp} ...")
    try:
        result = transcribe_whisperx(audio_fp, whisper_model="large-v3", device=device)
        chunk_text = result["text"]
        # Store the transcript for later concatenation
        SESSIONS[session_id]["transcripts"].append(chunk_text)
    except Exception as e:
        print("Transcription error:", e)
        print(traceback.format_exc())
        return {"error": str(e), "session_id": session_id}

    # Check if finalization is requested on this upload
    finalize_now = str(is_last).lower() in {"1", "true", "yes", "y"}
    if finalize_now:
        try:
            final_text, extraction = await finalize_session(session_id)
            # if evaluate:
            return {
                "session_id": session_id,
                "transcript": final_text,
                "pep_risk": simple_pep_rule(final_text, extraction),
                "extraction": extraction,
                "finalized": True,
            }
        except Exception as e:
            print("Finalize error:", e)
            print(traceback.format_exc())
            return {"error": str(e), "session_id": session_id}

    # Not final yet: return chunk transcript only
    return {"session_id": session_id, "transcript": chunk_text, "pep_risk": None, "finalized": False}


@app.post("/process_local")
async def process_local(
    filename: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    """Process a full-length prerecorded audio file from pep_risk/recordings.
    Transcribes the entire file once, runs ERCP extraction, saves CSV, and returns results.
    """
    audio_fp = os.path.join(RECORDINGS_DIR, filename)
    if not os.path.exists(audio_fp):
        return {"error": f"File not found: {audio_fp}"}

    # Ensure a session exists; use filename if not
    if not session_id:
        session_id = os.path.splitext(filename)[0]
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"chunks": [], "transcripts": [], "finalized": False, "final_transcript": None, "extraction": None}

    # Transcribe the full local file
    print(f"Transcribing local file {audio_fp} ...")
    try:
        result = transcribe_whisperx(audio_fp, whisper_model="large-v3", device=device)
        text = result.get("text", "")
    except Exception as e:
        print("Local transcription error:", e)
        print(traceback.format_exc())
        return {"error": str(e)}

    # Store transcript then finalize (concatenate -> LLM -> save CSV)
    SESSIONS[session_id]["transcripts"].append(text)
    final_text, extraction = await finalize_session(session_id)
    
    # Evaluate against ground truth (if available)
    eval_csv = os.path.join(RESULTS_DIR, "pep_eval.csv")
    eval_metrics = evaluate_against_ground_truth(
        original_filename=filename,
        extraction=extraction,
        ground_truth_csv=GROUND_TRUTH_CSV,
        save_to_csv=eval_csv
    )

    # Ensure evaluation metrics are JSON-serializable native types
    try:
        eval_metrics_safe = jsonable_encoder(eval_metrics)
    except Exception:
        # Fallback to returning a stringified version
        eval_metrics_safe = {"warning": "Failed to json-encode evaluation metrics"}

    return {
        "session_id": session_id,
        "transcript": final_text,
        "pep_risk": simple_pep_rule(final_text, extraction),
        "extraction": jsonable_encoder(extraction),
        "finalized": True,
        "evaluation": eval_metrics_safe,
    }
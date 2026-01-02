import traceback
from fastapi import FastAPI, UploadFile, Form
import os, uuid
from whisperx import load_model
try:
    import torch
    TORCH_AVAILABLE = True
    import torch.serialization
    import omegaconf
    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata,
        Any, Dict, List, Optional, list, dict, int, float, collections.defaultdict, omegaconf.nodes.AnyNode, omegaconf.base.Metadata, set, tuple, torch.torch_version.TorchVersion, 
    ])
    from pyannote.audio.core.model import Introspection
    from pyannote.audio.core.task import Specifications, Problem, Resolution
    torch.serialization.add_safe_globals([
        Introspection, Specifications, Problem, Resolution, torch.torch_version.TorchVersion, 
    ])
except Exception:
    torch = None
    TORCH_AVAILABLE = False
from typing import Any, Dict, List, Optional
from transcription.whisperx_transcribe import transcribe_whisperx
from llm.client import LLMClient
from processors.ercp_processor import ERCPProcessor
from pep_risk.peprisc_model import predict_pep_risk
import pandas as pd
from pep_risk.evaluation import evaluate_against_ground_truth
from fastapi.encoders import jsonable_encoder

app = FastAPI()

# Get absolute paths to avoid working directory issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "pep_risk", "results")
RECORDINGS_DIR = os.path.join(BASE_DIR, "pep_risk", "recordings") #!!change to exact dir
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "pep_risk", "results", "GROUND_TRUTH_LONG.csv")
BATCH_RESULTS_CSV = os.path.join(RESULTS_DIR, "long_pep_predictions.csv")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

#* Fix for PyTorch 2.6+ weights_only=True default
from typing import Dict, List, Optional
import collections

# Load WhisperX once at startup
if TORCH_AVAILABLE and getattr(torch, 'cuda', None) and torch.cuda.is_available():
    device = "cuda"
elif TORCH_AVAILABLE and hasattr(getattr(torch, 'backends', None), 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
try:
    model = load_model("large-v3", device=device)
except Exception as e:
    print(f"Warning: failed to load whisperx model at startup: {e}")
    model = None

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

ercp_processor = None
if llm_handler is not None:
    try:
        ercp_processor = ERCPProcessor(
            procedure_type="ercp",
            system_prompt_fp=os.path.join(BASE_DIR, "pep_risk", "prompts", "system.txt"),
            output_fp=os.path.join(BASE_DIR, "pep_risk", "results", "pep_server.csv"),
            llm_handler=llm_handler,
            to_postgres=False,
        )
    except Exception as e:
        print("Failed to initialize ERCPProcessor:", e)
        ercp_processor = None


async def process_pipeline(transcript: str) -> dict:
    if ercp_processor is None:
        raise RuntimeError("ERCP processor is not initialized; check LLM config and credentials.")
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Prepare the data row
    result_row = {
        "session_id": session_id,
        "transcript": final_transcript,
        **extraction  # Unpack all ERCP extraction fields
    }
    
    # Define the output CSV file path
    csv_path = os.path.join(RESULTS_DIR, "pep_llm_extraction.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved PEP risk extraction results to {csv_path}")


def load_manual_data_from_ground_truth(filename: str) -> Optional[dict]:
    """
    Load manual patient data from GROUND_TRUTH.csv based on audio filename.
    Maps ground truth columns to the format expected by the R model.
    
    Returns dict with manual fields or None if not found.
    """
    if not os.path.exists(GROUND_TRUTH_CSV):
        print(f"Ground truth file not found: {GROUND_TRUTH_CSV}")
        return None
    
    try:
        df = pd.read_csv(GROUND_TRUTH_CSV)
        
        # Try multiple matching strategies:
        # 1. Exact match on audio_recording column
        match = df[df['audio_recording'] == filename]
        # 2. on record_id (base filename without extension)
        if match.empty:
            base_filename = os.path.splitext(filename)[0]
            match = df[df['record_id'].astype(str) == base_filename]
        # 3. fuzzy match on audio_recording (contains base filename)
        if match.empty:
            base_filename = os.path.splitext(filename)[0]
            match = df[df['audio_recording'].astype(str).str.contains(base_filename, case=False, na=False)]
        if match.empty:
            print(f"No ground truth entry found for filename: {filename}")
            return None
        if len(match) > 1:
            print(f"Multiple ground truth matches for {filename}, using first match")
        
        row = match.iloc[0]
        
        # Map ground truth columns to R model field names
        manual_data = {
            "age_years": int(row['age']),
            "gender_male": bool(int(row['sex']) == 1),  # 1=male, 2=female
            "bmi": float(row['bmi']),
            "sod": bool(int(row['sod'])) if pd.notna(row['sod']) else False,
            "history_of_pep": bool(int(row['pep_history'])) if pd.notna(row['pep_history']) else False,
            "hx_of_recurrent_pancreatitis": bool(int(row['history_of_pancreatitis'])) if pd.notna(row['history_of_pancreatitis']) else False,
            "cholecystectomy": bool(int(row['cholecystectomy'])) if pd.notna(row['cholecystectomy']) else False,
            "trainee_involvement": bool(int(row['trainee_involvement'])) if pd.notna(row['trainee_involvement']) else False,
            "pancreo_biliary_malignancy": bool(int(row['malignancy'])) if pd.notna(row['malignancy']) else False,
            "indomethacin_nsaid_prophylaxis": bool(int(row['rectal_indo'])) if pd.notna(row['rectal_indo']) else False,
        }
        
        print(f"✓ Loaded manual data for {filename}: age={manual_data['age_years']}, gender_male={manual_data['gender_male']}, bmi={manual_data['bmi']}")
        return manual_data
        
    except Exception as e:
        print(f"Error loading manual data from ground truth: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_batch_prediction_results(filename: str, manual_data: dict, llm_extraction: dict, 
                                   prediction_result: dict, transcript: str):
    """
    Save complete batch processing results to CSV:
    - Manual patient data (from ground truth) - stored as manual_*
    - All LLM-extracted fields (both manual and procedural)
    - R model predictions for all 6 treatment scenarios
    """
    result_row = {
        "filename": filename,
    }
    
    # Add manual data from ground truth (used by R model) with manual_ prefix
    for key, value in manual_data.items():
        result_row[f"manual_{key}"] = value
    
    # ALL LLM-extracted fields
    for key, value in llm_extraction.items():
        if key not in ["id", "model"]:
            result_row[key] = value
    
    # R model prediction results
    result_row["prediction_success"] = prediction_result.get("success")
    
    # Treatment predictions
    treatment_predictions = prediction_result.get("treatment_predictions", [])
    for tp in treatment_predictions:
        therapy_name = tp.get("therapy_id") or tp.get("therapy") or "unknown"
        therapy_name = str(therapy_name).lower().replace(" ", "_").replace("+", "and")
        result_row[f"risk_{therapy_name}_pct"] = tp.get("risk_percentage")
    result_row["baseline_risk_pct"] = prediction_result.get("risk_score")

    result_row["transcript"] = transcript
    
    # Ensure results directory exists (use absolute path)
    os.makedirs(os.path.dirname(BATCH_RESULTS_CSV), exist_ok=True)
    
    # Load or create DataFrame
    if os.path.exists(BATCH_RESULTS_CSV):
        df = pd.read_csv(BATCH_RESULTS_CSV)
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    
    df.to_csv(BATCH_RESULTS_CSV, index=False)
    print(f"✓ Saved batch prediction results to {BATCH_RESULTS_CSV}")


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

    finalize_now = str(is_last).lower() in {"1", "true", "yes", "y"}
    if finalize_now:
        try:
            final_text, extraction = await finalize_session(session_id)
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
    """
    Process a full-length prerecorded audio file from pep_risk/recordings.
    
    Workflow:
    1. Transcribe audio
    2. Extract procedural risk factors via LLM
    3. Load manual patient data from GROUND_TRUTH.csv
    4. Combine manual + LLM data and send to R PEPRISC model
    5. Get predictions for all 6 treatment scenarios
    6. Save complete results to CSV
    7. Evaluate LLM extraction against ground truth
    """
    audio_fp = os.path.join(RECORDINGS_DIR, filename)
    if not os.path.exists(audio_fp):
        return {"error": f"File not found: {audio_fp}"}

    # Create a new session for each request (or reuse if explicitly provided)
    if not session_id:
        session_id = f"{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:8]}"
    
    # Check if session already exists and was finalized - start fresh
    if session_id in SESSIONS and SESSIONS[session_id].get("finalized"):
        print(f"Session {session_id} already finalized, creating new session")
        session_id = f"{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:8]}"
    
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"chunks": [], "transcripts": [], "finalized": False, "final_transcript": None, "extraction": None}

    print(f"\n{'='*80}")
    print(f"Processing: {filename}")
    print(f"{'='*80}")
    
    # Step 1: Transcribe the full local file
    print(f"[1/6] Transcribing audio file {audio_fp}...")
    try:
        result = transcribe_whisperx(audio_fp, whisper_model="large-v3", device=device)
        text = result.get("text", "")
        print(f"✓ Transcription complete: {len(text)} characters")
    except Exception as e:
        print(f"✗ Transcription error: {e}")
        print(traceback.format_exc())
        return {"error": str(e), "step": "transcription"}

    # Step 2: Extract procedural risk factors via LLM
    print(f"[2/6] Extracting PEP risk factors via LLM...")
    SESSIONS[session_id]["transcripts"].append(text)
    try:
        final_text, extraction = await finalize_session(session_id)
        print(f"✓ LLM extraction complete: {len(extraction)} fields extracted")
    except Exception as e:
        print(f"✗ LLM extraction error: {e}")
        print(traceback.format_exc())
        return {"error": str(e), "step": "llm_extraction"}
    
    # Step 3: Load manual patient data from ground truth
    print(f"[3/6] Loading manual patient data from ground truth...")
    manual_data = load_manual_data_from_ground_truth(filename)
    if manual_data is None:
        print(f"Warning: No manual data found for {filename}, using defaults")
        manual_data = {}
    
    # Step 4: Send to R PEPRISC model for prediction
    print(f"[4/6] Running R PEPRISC model prediction...")
    try:
        prediction_result = predict_pep_risk(
            manual_data=manual_data,
            llm_extracted_data=extraction
        )
        
        if prediction_result.get("success"):
            treatment_count = len(prediction_result.get("treatment_predictions", []))
            baseline_risk = prediction_result.get("risk_score")
            print(f"✓ R model prediction success:")
            print(f"  - Baseline risk: {baseline_risk}% ({prediction_result.get('risk_category')})")
            print(f"  - Treatment scenarios: {treatment_count}")
        else:
            print(f"✗ R model prediction failed: {prediction_result.get('error', 'Unknown error')}")
            prediction_result = {"success": False, "error": "R model prediction failed"}
    except Exception as e:
        print(f"✗ R model error: {e}")
        print(traceback.format_exc())
        prediction_result = {"success": False, "error": str(e)}
    
    print(f"[5/6] Saving batch results to CSV...")
    try:
        save_batch_prediction_results(
            filename=filename,
            manual_data=manual_data,
            llm_extraction=extraction,
            prediction_result=prediction_result,
            transcript=final_text
        )
    except Exception as e:
        print(f"Failed to save batch results: {e}")
    
    print(f"[6/6] Evaluating LLM extraction against ground truth...")
    eval_csv = os.path.join(RESULTS_DIR, "pep_eval.csv")
    try:
        eval_metrics = evaluate_against_ground_truth(
            original_filename=filename,
            extraction=extraction,
            ground_truth_csv=GROUND_TRUTH_CSV,
            save_to_csv=eval_csv
        )
        eval_metrics_safe = jsonable_encoder(eval_metrics)
        print(f"Evaluation complete")
    except Exception as e:
        print(f"Fail evaluation: {e}")
        eval_metrics_safe = {"error": str(e)}

    print(f"{'='*80}")
    print(f"✓ Processing complete for {filename}")
    print(f"{'='*80}\n")

    # Return comprehensive results
    return {
        "session_id": session_id,
        "filename": filename,
        "transcript": final_text,
        "extraction": jsonable_encoder(extraction),
        "manual_data": manual_data,
        "prediction": jsonable_encoder(prediction_result),
        "evaluation": eval_metrics_safe,
        "finalized": True,
    }
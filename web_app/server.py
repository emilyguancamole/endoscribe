import os
import sys
import uuid
import json
import time
import asyncio
import traceback
import warnings
import signal
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Filter external library warnings that we can't control
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain.utils.torch_audio_backend")

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from web_app.models import (
    ProcessRequest,
    ProcessResponse,
    HealthResponse,
    ProcedureType,
    WebSocketMessage
)
from llm.llm_client import LLMClient
from processors import ColProcessor, ERCPProcessor, EUSProcessor, EGDProcessor
from transcription.whisperx_transcribe import transcribe_whisperx
import whisperx

# Setup directories
BASE_DIR = Path(__file__).parent

# Use persistent volumes in production (Fly.io), local dirs in development
if os.getenv("FLY_APP_NAME"):
    # Production: use persistent volume
    UPLOAD_DIR = Path("/data/uploads")
    RESULTS_DIR = Path("/data/results")
    MODELS_DIR = Path("/data/models")
else:
    # Development: use local directories
    UPLOAD_DIR = BASE_DIR / "uploads"
    RESULTS_DIR = BASE_DIR / "results"
    MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Global state
WHISPER_MODEL = None
WHISPER_ALIGN_MODEL = None
WHISPER_ALIGN_METADATA = None
WHISPER_DEVICE = None  # Actual device WhisperX is using (cpu for MPS, cuda for CUDA)
LLM_HANDLER = None
PROCESSOR_MAP = {}
SESSIONS: Dict[str, Dict] = {}

# Scale-to-zero: Idle timeout configuration
# Enable on Fly.io to reduce GPU costs; disable locally for development
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "60"))  # Default: 60 seconds
ENABLE_IDLE_SHUTDOWN = os.getenv("FLY_APP_NAME") is not None  # Only on Fly.io
last_activity_time = time.time()
idle_check_task = None

# Device configuration with detailed diagnostics
# Priority: CUDA (Fly.io/Linux) > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"\n{'='*60}")
print(f"GPU DIAGNOSTICS")
print(f"{'='*60}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, "mps"):
    print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Using device: {DEVICE}")

if DEVICE == "cuda":
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
elif DEVICE == "mps":
    print(f"Running on Apple Silicon GPU (MPS)")
    print(f"Note: MPS provides ~2-3x speedup over CPU on M1/M2/M3 Macs")
else:
    print("WARNING: No GPU acceleration! Transcription will be SLOW on CPU.")
    print("For faster performance:")
    print("  - Production: Use Fly.io with A10 GPU")
    print("  - Mac: Ensure PyTorch with MPS support is installed")
print(f"{'='*60}\n")


# Scale-to-zero helper functions
def update_activity():
    """Update last activity timestamp"""
    global last_activity_time
    last_activity_time = time.time()


async def check_idle_and_shutdown():
    """Background task that checks for idle time and shuts down if necessary"""
    global idle_check_task

    if not ENABLE_IDLE_SHUTDOWN:
        print("Idle shutdown disabled (not on Fly.io)")
        return

    print(f"Idle shutdown enabled: will exit after {IDLE_TIMEOUT_SECONDS}s of inactivity")

    while True:
        await asyncio.sleep(10)  # Check every 10 seconds

        idle_time = time.time() - last_activity_time

        if idle_time >= IDLE_TIMEOUT_SECONDS:
            print(f"\n{'='*60}")
            print(f"IDLE SHUTDOWN TRIGGERED")
            print(f"{'='*60}")
            print(f"Idle time: {idle_time:.1f}s (threshold: {IDLE_TIMEOUT_SECONDS}s)")
            print(f"Shutting down to save GPU costs...")
            print(f"Fly Proxy will restart on next request")
            print(f"{'='*60}\n")

            # Graceful shutdown
            os.kill(os.getpid(), signal.SIGTERM)
            break


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup and cleanup on shutdown"""
    global WHISPER_MODEL, WHISPER_ALIGN_MODEL, WHISPER_ALIGN_METADATA, WHISPER_DEVICE, LLM_HANDLER, PROCESSOR_MAP, idle_check_task

    # Startup
    print("Initializing WhisperX model...")
    try:
        # Use appropriate compute type based on device
        # WhisperX doesn't support MPS directly, so fall back to CPU for MPS
        WHISPER_DEVICE = DEVICE if DEVICE == "cuda" else "cpu"

        if DEVICE == "cuda":
            compute_type = "float16"
        elif DEVICE == "mps":
            compute_type = "int8"
            print("Note: WhisperX doesn't support MPS directly, using CPU with int8")
            print("For best performance on Mac, consider using faster-whisper directly")
        else:
            compute_type = "int8"

        print(f"Loading WhisperX with device={WHISPER_DEVICE}, compute_type={compute_type}")
        print("Note: First-time download of large-v3 model (~5GB) may take several minutes...")
        WHISPER_MODEL = whisperx.load_model("large-v3", WHISPER_DEVICE, compute_type=compute_type)
        print("WhisperX model loaded successfully!")

        # Load alignment model once during startup
        print("Loading WhisperX alignment model...")
        WHISPER_ALIGN_MODEL, WHISPER_ALIGN_METADATA = whisperx.load_align_model(
            language_code="en",
            device=WHISPER_DEVICE
        )
        print("WhisperX alignment model loaded successfully!")
    except Exception as e:
        print(f"Failed to load WhisperX model: {e}")
        import traceback
        traceback.print_exc()
        WHISPER_MODEL = None
        WHISPER_ALIGN_MODEL = None
        WHISPER_ALIGN_METADATA = None

    print("Initializing LLM handler...")
    try:
        # Use anthropic_claude config as specified in your command
        LLM_HANDLER = LLMClient.from_config("anthropic_claude")
        print("LLM handler initialized successfully")
    except Exception as e:
        print(f"Failed to initialize LLM handler: {e}")
        LLM_HANDLER = None

    # Initialize processors for all procedure types
    if LLM_HANDLER:
        print("Initializing processors...")
        try:
            PROCESSOR_MAP = {
                "col": ColProcessor(
                    procedure_type="col",
                    system_prompt_fp="prompts/col/system.txt",
                    output_fp=str(RESULTS_DIR / "col_results.csv"),
                    llm_handler=LLM_HANDLER,
                    to_postgres=False
                ),
                "eus": EUSProcessor(
                    procedure_type="eus",
                    system_prompt_fp="prompts/eus/system.txt",
                    output_fp=str(RESULTS_DIR / "eus_results.csv"),
                    llm_handler=LLM_HANDLER,
                    to_postgres=False
                ),
                "ercp": ERCPProcessor(
                    procedure_type="ercp",
                    system_prompt_fp="prompts/ercp/system.txt",
                    output_fp=str(RESULTS_DIR / "ercp_results.csv"),
                    llm_handler=LLM_HANDLER,
                    to_postgres=False
                ),
                "egd": EGDProcessor(
                    procedure_type="egd",
                    system_prompt_fp="prompts/egd/system.txt",
                    output_fp=str(RESULTS_DIR / "egd_results.csv"),
                    llm_handler=LLM_HANDLER,
                    to_postgres=False
                ),
            }
            print("All processors initialized successfully")
        except Exception as e:
            print(f"Failed to initialize processors: {e}")
            traceback.print_exc()

    # Start idle shutdown checker (only on Fly.io)
    if ENABLE_IDLE_SHUTDOWN:
        print(f"\nStarting idle shutdown monitor ({IDLE_TIMEOUT_SECONDS}s timeout)...")
        idle_check_task = asyncio.create_task(check_idle_and_shutdown())

    yield  # Application runs here

    # Shutdown (cleanup if needed)
    print("Shutting down...")

    # Cancel idle check task if running
    if idle_check_task and not idle_check_task.done():
        idle_check_task.cancel()
        try:
            await idle_check_task
        except asyncio.CancelledError:
            pass


# Initialize FastAPI app with lifespan handler
app = FastAPI(title="EndoScribe Web API", version="1.0.0", lifespan=lifespan)

# Setup templates and static files
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI"""
    update_activity()
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check if volumes are accessible in production
    volumes_ok = True
    if os.getenv("FLY_APP_NAME"):
        volumes_ok = (
            Path("/data").exists() and
            Path("/data/uploads").exists() and
            Path("/data/results").exists()
        )

    status = "healthy" if (WHISPER_MODEL and LLM_HANDLER and volumes_ok) else "degraded"

    return HealthResponse(
        status=status,
        whisper_loaded=WHISPER_MODEL is not None,
        llm_initialized=LLM_HANDLER is not None,
        supported_procedures=["col", "eus", "ercp", "egd"]
    )


@app.get("/gpu-info")
async def gpu_info():
    """GPU diagnostics endpoint"""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device": DEVICE,
        "whisperx_device": WHISPER_DEVICE,  # Actual device WhisperX is using
    }

    if DEVICE == "cuda":
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            "gpu_count": torch.cuda.device_count(),
        })
    elif DEVICE == "mps":
        info.update({
            "platform": "Apple Silicon",
            "note": "WhisperX uses CPU on MPS (MPS not directly supported by WhisperX)",
            "recommendation": "Use faster-whisper for native MPS support on Mac"
        })

    return info


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription

    Protocol:
    1. Client sends JSON: {"type": "start", "session_id": "optional"}
    2. Client sends binary audio chunks (WebM, WAV, etc.)
    3. Server transcribes and sends back: {"type": "transcript", "data": {"text": "...", "session_id": "..."}}
    4. Client sends JSON: {"type": "end"} to finalize
    """
    await websocket.accept()
    update_activity()  # Track WebSocket connection for idle shutdown
    session_id = None
    audio_chunks = []

    try:
        while True:
            # Receive data (can be text or bytes)
            try:
                data = await websocket.receive()
                update_activity()  # Track activity on every message
            except WebSocketDisconnect:
                print(f"WebSocket disconnected for session {session_id}")
                break

            # Handle text messages (control messages)
            if "text" in data:
                try:
                    message = json.loads(data["text"])
                    msg_type = message.get("type")

                    if msg_type == "start":
                        session_id = message.get("session_id") or str(uuid.uuid4())
                        SESSIONS[session_id] = {
                            "chunks": [],
                            "transcripts": [],
                            "started_at": datetime.now()
                        }
                        await websocket.send_json({
                            "type": "status",
                            "message": "Session started",
                            "session_id": session_id
                        })
                        print(f"Started session {session_id}")

                    elif msg_type == "end":
                        await websocket.send_json({
                            "type": "status",
                            "message": "Session ended",
                            "session_id": session_id
                        })
                        print(f"Ended session {session_id}")
                        break

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })

            # Handle binary audio data
            elif "bytes" in data:
                if not session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No active session. Send 'start' message first."
                    })
                    continue

                if WHISPER_MODEL is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Whisper model not initialized"
                    })
                    continue

                audio_data = data["bytes"]
                print(f"Received {len(audio_data)} bytes of audio for session {session_id}")

                # Save audio chunk to temporary file
                chunk_id = str(uuid.uuid4())
                audio_path = UPLOAD_DIR / f"{session_id}_{chunk_id}.webm"

                with open(audio_path, "wb") as f:
                    f.write(audio_data)

                SESSIONS[session_id]["chunks"].append(str(audio_path))

                # Transcribe the audio chunk
                try:
                    print(f"Transcribing audio chunk {chunk_id}...")

                    # Send processing status to client
                    await websocket.send_json({
                        "type": "status",
                        "message": "Processing audio chunk...",
                        "session_id": session_id
                    })

                    # Use the pre-loaded global model instead of loading a new one
                    if WHISPER_MODEL is None:
                        raise Exception("WhisperX model not initialized")

                    # Define transcription function to run in thread
                    def transcribe_audio():
                        import whisperx
                        audio = whisperx.load_audio(str(audio_path))
                        # Use larger batch size on GPU for better performance
                        if DEVICE == "cuda":
                            batch_size = 16  # Large batch for NVIDIA GPU
                        elif DEVICE == "mps":
                            batch_size = 8   # Medium batch for Apple Silicon
                        else:
                            batch_size = 4   # Small batch for CPU
                        print(f"Transcribing with batch_size={batch_size} on {DEVICE}")
                        result_raw = WHISPER_MODEL.transcribe(audio, batch_size=batch_size, language="en")

                        # Align for better accuracy using cached alignment model
                        if WHISPER_ALIGN_MODEL is not None and WHISPER_ALIGN_METADATA is not None:
                            aligned_result = whisperx.align(
                                result_raw["segments"],
                                WHISPER_ALIGN_MODEL,
                                WHISPER_ALIGN_METADATA,
                                audio,
                                WHISPER_DEVICE,  # Use WhisperX device (cpu for MPS, cuda for CUDA)
                                return_char_alignments=False
                            )
                            return aligned_result["segments"]
                        else:
                            # Fallback to unaligned if alignment model not available
                            print("Warning: Using unaligned transcription (alignment model not loaded)")
                            return result_raw["segments"]

                    # Run transcription in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    segments = await loop.run_in_executor(None, transcribe_audio)

                    # Join segments into text
                    transcript_text = " ".join(seg["text"] for seg in segments).replace("  ", " ").strip()

                    SESSIONS[session_id]["transcripts"].append(transcript_text)

                    # Send transcription back to client
                    await websocket.send_json({
                        "type": "transcript",
                        "data": {
                            "text": transcript_text,
                            "session_id": session_id,
                            "chunk_id": chunk_id,
                            "timestamp": time.time()
                        }
                    })
                    print(f"Sent transcript: {transcript_text[:100]}...")

                    # Send status update to clear processing state
                    await websocket.send_json({
                        "type": "status",
                        "message": "Transcription complete",
                        "session_id": session_id
                    })

                except Exception as e:
                    error_msg = f"Transcription error: {str(e)}"
                    print(error_msg)
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "message": error_msg,
                        "session_id": session_id
                    })

    except Exception as e:
        error_msg = f"WebSocket error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": error_msg
            })
        except:
            pass

    finally:
        # Cleanup
        if session_id and session_id in SESSIONS:
            print(f"Cleaning up session {session_id}")
            # Optionally delete temporary audio files
            # for chunk_path in SESSIONS[session_id]["chunks"]:
            #     try:
            #         os.remove(chunk_path)
            #     except:
            #         pass


@app.post("/api/process", response_model=ProcessResponse)
async def process_transcript(request: ProcessRequest):
    """
    Process a transcript and extract structured data

    Args:
        request: ProcessRequest with transcript, procedure_type, and optional session_id

    Returns:
        ProcessResponse with extracted structured data
    """
    update_activity()  # Track API activity for idle shutdown
    start_time = time.time()

    # Validate LLM is initialized
    if LLM_HANDLER is None:
        raise HTTPException(status_code=503, detail="LLM handler not initialized")

    # Get processor for procedure type
    processor = PROCESSOR_MAP.get(request.procedure_type.value)
    if not processor:
        raise HTTPException(
            status_code=400,
            detail=f"Processor not available for procedure type: {request.procedure_type}"
        )

    try:
        # Build messages for LLM
        import pandas as pd

        # Create a single-row dataframe with the transcript
        transcript_df = pd.DataFrame([{
            "participant_id": request.session_id or "web_session",
            "pred_transcript": request.transcript
        }])

        # Process based on procedure type
        if request.procedure_type == ProcedureType.COL:
            # Colonoscopy processing (returns both colonoscopy and polyp data)
            col_outputs = []
            polyp_outputs = []

            for _, row in transcript_df.iterrows():
                # Colonoscopy-level processing
                col_messages = processor.build_messages(
                    row["pred_transcript"],
                    system_prompt_fp=processor.system_prompt_fp,
                    prompt_field_definitions_fp='./prompts/col/colonoscopies.txt',
                    fewshot_examples_dir="./prompts/col/fewshot",
                    prefix="col"
                )

                if LLM_HANDLER.model_type in ["openai", "anthropic"]:
                    col_response = LLM_HANDLER.chat(col_messages)
                else:
                    col_response = LLM_HANDLER.chat(col_messages)[0].outputs[0].text.strip()

                # Parse JSON response
                # TODO: Remove this manual JSON validation once LLM is returning proper JSON
                print(f"Col response: {col_response[:500]}")  # Log first 500 chars
                start_idx = col_response.find("{")
                end_idx = col_response.rfind("}")

                if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                    raise ValueError(f"No valid JSON found in response. Response: {col_response[:500]}")

                json_str = col_response[start_idx:end_idx + 1]
                col_json = json.loads(json_str)
                col_data = processor.parse_validate_colonoscopy_response(col_json, row["participant_id"])
                col_outputs.append(col_data)

                # Polyp-level processing
                polyp_messages = processor.build_polyp_messages(
                    row["pred_transcript"],
                    findings=col_json.get("findings", ""),
                    polyp_count=col_json.get("polyp_count", 0),
                    system_prompt_fp=processor.system_prompt_fp
                )

                if LLM_HANDLER.model_type in ["openai", "anthropic"]:
                    polyp_response = LLM_HANDLER.chat(polyp_messages)
                else:
                    polyp_response = LLM_HANDLER.chat(polyp_messages)[0].outputs[0].text.strip()

                # Extract JSON array from response (handle cases where LLM adds explanatory text)
                # TODO: Remove this manual JSON validation once LLM is returning proper JSON
                print(f"Polyp response: {polyp_response[:500]}")  # Log first 500 chars
                start_idx = polyp_response.find("[")
                end_idx = polyp_response.rfind("]")

                if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                    raise ValueError(f"No valid JSON array found in polyp response. Response: {polyp_response[:500]}")

                json_str = polyp_response[start_idx:end_idx + 1]
                polyps_json = json.loads(json_str)
                polyp_data = processor.parse_validate_polyp_response(polyps_json, row["participant_id"])
                polyp_outputs.extend(polyp_data)

            result_data = {
                "colonoscopy": col_outputs[0] if col_outputs else {},
                "polyps": polyp_outputs
            }

        else:
            # Other procedure types (EUS, ERCP, EGD)
            outputs = []

            for _, row in transcript_df.iterrows():
                # Determine the correct prompt field definitions file
                prompt_files = {
                    "eus": "./prompts/eus/eus.txt",
                    "ercp": "./prompts/ercp/ercp.txt",
                    "egd": "./prompts/egd/egd.txt"
                }

                messages = processor.build_messages(
                    row["pred_transcript"],
                    system_prompt_fp=processor.system_prompt_fp,
                    prompt_field_definitions_fp=prompt_files.get(request.procedure_type.value, ""),
                    fewshot_examples_dir=f"./prompts/{request.procedure_type.value}/fewshot",
                    prefix=request.procedure_type.value
                )

                if LLM_HANDLER.model_type in ["openai", "anthropic"]:
                    response = LLM_HANDLER.chat(messages)
                else:
                    response = LLM_HANDLER.chat(messages)[0].outputs[0].text.strip()

                # Parse JSON response
                response_json = json.loads(response[response.find("{"): response.rfind("}") + 1])
                outputs.append({
                    "id": row["participant_id"],
                    "model": LLM_HANDLER.model_type,
                    **response_json
                })

            result_data = outputs[0] if outputs else {}

        processing_time = time.time() - start_time

        return ProcessResponse(
            success=True,
            procedure_type=request.procedure_type.value,
            session_id=request.session_id,
            data=result_data,
            processing_time_seconds=round(processing_time, 2)
        )

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        return ProcessResponse(
            success=False,
            procedure_type=request.procedure_type.value,
            session_id=request.session_id,
            error=error_msg,
            processing_time_seconds=round(time.time() - start_time, 2)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

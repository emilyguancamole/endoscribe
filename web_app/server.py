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

try:
    import torch
    import torch.serialization
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from typing import Any, Dict, List, Tuple, Optional
import collections

if TORCH_AVAILABLE:
    import omegaconf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    try:
        torch.serialization.add_safe_globals([
            ListConfig, DictConfig, ContainerMetadata,
            Any, Dict, List, Tuple, Optional, list, dict, collections.defaultdict, int, float, omegaconf.nodes.AnyNode, omegaconf.base.Metadata, set, tuple, torch.torch_version.TorchVersion,
        ])
    except Exception:
        # Best-effort: if adding safe globals fails, continue without crashing
        pass


def _convert_bytes_to_pcm16le_16k(input_bytes: bytes) -> bytes:
    """
    Convert an audio blob (webm/ogg/mp3/etc) in bytes to raw PCM16LE 16k mono using ffmpeg.
    This is a blocking call and intended to be run in a thread executor via asyncio to avoid blocking the event loop.
    Returns raw PCM bytes to write to Azure PushAudioInputStream.
    """
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", "-", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", "16000", "-"
        ]
        proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")
        return proc.stdout
    except Exception as e:
        print(f"Error converting audio bytes to PCM: {e}")
        raise

# External library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain.utils.torch_audio_backend")

# Project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from web_app.models import (
    ProcessRequest,
    ProcessResponse,
    HealthResponse,
    ProcedureType,
)
from pydantic import BaseModel
from llm.llm_client import LLMClient
from processors import ColProcessor, ERCPProcessor, EUSProcessor, EGDProcessor
from transcription.transcription_service import TranscriptionConfig, transcribe_unified
from data_models.data_models import (
    ColonoscopyData,
    PolypData,
    EUSData,
    EGDData,
    PEPRiskData
)
from data_models.generated_ercp_base_model import ErcpBaseData
from pydantic import ValidationError
from web_app.config import CONFIG

BASE_DIR = Path(__file__).parent
# Use persistent volumes in production (Fly.io), local in dev
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

# Session storage (json) so saved notes survive process restarts
SESSIONS_STORE_DIR = RESULTS_DIR / "sessions"
SESSIONS_STORE_DIR.mkdir(parents=True, exist_ok=True)

def _session_store_path(session_id: str) -> Path:
    return SESSIONS_STORE_DIR / f"{session_id}.json"

def _persist_session_record(session_id: str, record: Dict[str, Any]) -> None:
    """Atomically persist a session record to disk."""
    fp = _session_store_path(session_id)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    record = dict(record)
    record["session_id"] = session_id
    record["updated_at"] = datetime.now().isoformat()
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, fp)

def _load_persisted_session_record(session_id: str) -> Optional[Dict[str, Any]]:
    fp = _session_store_path(session_id)
    if not fp.exists():
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load session {session_id} from disk: {e}")
        return None

def _list_persisted_session_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        for fp in SESSIONS_STORE_DIR.glob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                    if isinstance(rec, dict) and rec.get("session_id"):
                        records.append(rec)
            except Exception:
                continue
    except Exception as e:
        print(f"Failed to list persisted sessions: {e}")
    return records

# Transcription configuration ('azure', 'whisperx')
TRANSCRIPTION_CONFIG = TranscriptionConfig()
# Allow explicit override otherwise rely on TranscriptionConfig auto-det
if os.getenv("TRANSCRIPTION_SERVICE"):
    TRANSCRIPTION_CONFIG.service = os.getenv("TRANSCRIPTION_SERVICE")
if TRANSCRIPTION_CONFIG.use_azure():
    print("Configured transcription service: azure (AZURE_SPEECH_KEY detected)")
else:
    print(f"Configured transcription service: {TRANSCRIPTION_CONFIG.service}")

# Global state
WHISPER_MODEL = None
WHISPER_ALIGN_MODEL = None
WHISPER_ALIGN_METADATA = None
WHISPER_DEVICE = None  # Actual device WhisperX is using (cpu for MPS, cuda for CUDA)
LLM_HANDLER = None
PROCESSOR_MAP = {}
SESSIONS: Dict[str, Dict] = {}

last_activity_time = time.time()
idle_check_task = None

class SaveSessionRequest(BaseModel):
    note_content: str
    procedure_type: Optional[str] = None
    transcript: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

# Device configuration with detailed diagnostics
if TORCH_AVAILABLE and getattr(torch, 'cuda', None) and torch.cuda.is_available():
    DEVICE = "cuda"
elif TORCH_AVAILABLE and hasattr(getattr(torch, 'backends', None), "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"\n{'='*60}")
print("GPU DIAGNOSTICS")
if TORCH_AVAILABLE:
    try:
        print(f"PyTorch version: {torch.__version__}")
        cuda_avail = getattr(torch.cuda, 'is_available', lambda: False)()
        print(f"CUDA available: {cuda_avail}")
    except Exception:
        print("PyTorch present but diagnostics failed")
else:
    print("PyTorch not installed; running in CPU-only mode")
print(f"Using device: {DEVICE}")

async def check_idle_and_shutdown():
    """Monitor idle time and trigger shutdown if no activity detected."""
    global last_activity_time
    try:
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            # If there are active WebSocket sessions, app active and skip shutdown. 
            if SESSIONS: # refresh last_activity_time so that the idle timer doesn't fire
                last_activity_time = time.time()
                continue

            idle_duration = time.time() - last_activity_time
            if idle_duration > CONFIG.IDLE_TIMEOUT_SECONDS:
                print(f"\nNo activity for {idle_duration:.0f}s (timeout: {CONFIG.IDLE_TIMEOUT_SECONDS}s)")
                print("Triggering graceful shutdown to scale to zero...")
                # graceful shutdown via SIGTERM so FastAPI/uvicorn can cleanup
                os.kill(os.getpid(), signal.SIGTERM)
                break
    except asyncio.CancelledError: # Task cancelled during shutdown - ignore
        pass


def concatenate_audio_chunks(chunk_paths, output_path):
    """
    Concatenate multiple audio chunks into a single audio file.
    From the pep_risk approach where complete audio files are transcribed rather than tiny chunks. 
    Uses pydub to concatenate webm/wav files.
    
    Args:
        chunk_paths: List of paths to audio chunks
        output_path: Path to save concatenated audio
    Returns:
        Path to concatenated audio file
    """
    try:
        from pydub import AudioSegment
        if not chunk_paths:
            raise ValueError("No audio chunks to concatenate")
        
        print(f"Loading first chunk: {chunk_paths[0]}")
        combined = AudioSegment.from_file(chunk_paths[0])
        print(f"  Duration: {len(combined)}ms, Channels: {combined.channels}")
        
        # Concatenate remaining chunks
        for chunk_path in chunk_paths[1:]:
            audio = AudioSegment.from_file(chunk_path)
            combined += audio
        
        # proper format for speech recognition
        combined = combined.set_channels(1)  # Mono
        combined = combined.set_frame_rate(16000)  # 16kHz
        combined = combined.set_sample_width(2)  # 16-bit
        combined.export(output_path, format="wav") # wav
        duration_sec = len(combined) / 1000.0
        print(f"Concatenated {len(chunk_paths)} chunks into {output_path}")
        
        # Quick diagnostic: check if audio is actually silent
        if combined.dBFS < -60:
            print(f"  WARNING: Audio appears very quiet (dBFS: {combined.dBFS:.1f}). Microphone may not be working.")
        
        return output_path
        
    except Exception as e:
        print(f"Error concatenating audio chunks: {e}")
        traceback.print_exc()
        return chunk_paths[0] if chunk_paths else None # Fallback: return first chunk


def transcribe_audio(audio_path: str, service: Optional[str] = None, **kwargs):
    """
    Unified transcription helper for the web app.

    Calls `transcription.transcription_service.transcribe_unified` under the hood and
    returns a single transcription dict (same shape as previous WhisperX helper):
    {"text": str, "segments": list, ...}

    Parameters:
    - audio_path: Path to the audio file to transcribe
    - service: Optional override for transcription service ('azure' or 'whisperx')
    - kwargs: forwarded to `transcribe_unified` (e.g., device, whisper_model)
    """
    svc = service or TRANSCRIPTION_CONFIG.service
    # transcribe_unified expects a list of files
    results = transcribe_unified(
        audio_files=[audio_path],
        service=svc,
        save=False,
        **kwargs
    )
    if isinstance(results, list) and len(results) > 0:
        return results[0]
    # Fallback: return empty structure
    return {"text": "", "segments": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup and cleanup on shutdown"""
    global WHISPER_MODEL, WHISPER_ALIGN_MODEL, WHISPER_ALIGN_METADATA, WHISPER_DEVICE, LLM_HANDLER, PROCESSOR_MAP, idle_check_task

    # Startup
    # Initialize WhisperX only if configured to use WhisperX; otherwise skip
    if TRANSCRIPTION_CONFIG.use_whisperx():
        print("Initializing WhisperX model...")
        try:
            import whisperx
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

            #! Fix for PyTorch 2.6+ weights_only=True default - for whisperx
            
            if TORCH_AVAILABLE:
                from pyannote.audio.core.model import Introspection
                from pyannote.audio.core.task import Specifications, Problem, Resolution
                try:
                    torch.serialization.add_safe_globals([
                        Introspection, Specifications, Problem, Resolution, torch.torch_version.TorchVersion,
                    ])
                except Exception:
                    pass

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
        llm_config = os.getenv("LLM_CONFIG", "openai_gpt4o")
        print(f"Using LLM config: {llm_config}")
        LLM_HANDLER = LLMClient.from_config(llm_config)
        print("LLM handler initialized successfully")
    except Exception as e:
        print(f"Failed to initialize LLM handler: {e}")
        traceback.print_exc()
        LLM_HANDLER = None

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
                "pep_risk": ERCPProcessor(
                    procedure_type="pep_risk",
                    system_prompt_fp="pep_risk/prompts/system.txt",
                    output_fp=str(RESULTS_DIR / "pep_risk_results.csv"),
                    llm_handler=LLM_HANDLER,
                    to_postgres=False
                ),
            }
            print("All processors initialized successfully")
        except Exception as e:
            print(f"Failed to initialize processors: {e}")
            traceback.print_exc()

    # Start idle shutdown checker (only on Fly.io)
    # if CONFIG.ENABLE_IDLE_SHUTDOWN:
    #     print(f"\nStarting idle shutdown monitor ({CONFIG.IDLE_TIMEOUT_SECONDS}s timeout)")
    #     idle_check_task = asyncio.create_task(check_idle_and_shutdown())

    yield  # Application runs here

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


@app.middleware("http")
async def update_last_activity_http(request: Request, call_next):
    """Update `last_activity_time` on every incoming HTTP request to indicate activity.

    This helps the idle shutdown watcher know when the app is receiving traffic.
    """
    global last_activity_time
    last_activity_time = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        last_activity_time = time.time()

# Setup static files - serve React build
REACT_BUILD_DIR = BASE_DIR / "static" / "dist"
if REACT_BUILD_DIR.exists():
    # Serve React build assets
    app.mount("/assets", StaticFiles(directory=str(REACT_BUILD_DIR / "assets")), name="assets")
    # Keep legacy static for favicon and other assets
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
else:
    # Fallback to legacy static for development
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Keep templates for fallback
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI - React build or fallback to legacy"""
    react_index = REACT_BUILD_DIR / "index.html"
    if react_index.exists():
        return FileResponse(str(react_index))
    else:
        # Fallback to legacy template
        return templates.TemplateResponse("index.html", {"request": request})


# Serve favicon at root so React build's `/favicon.svg` works when deployed
FAV_DIST = REACT_BUILD_DIR / "favicon.svg"
FAV_LEGACY = BASE_DIR / "static" / "favicon.svg"


@app.get("/favicon.svg")
async def favicon_svg():
    if REACT_BUILD_DIR.exists() and FAV_DIST.exists():
        return FileResponse(str(FAV_DIST))
    if FAV_LEGACY.exists():
        return FileResponse(str(FAV_LEGACY))
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/favicon.ico")
async def favicon_ico():
    # Some clients request /favicon.ico by default — return the svg with correct media type if no .ico present
    ico_path = None
    # prefer an actual .ico if present in the dist or legacy static
    if REACT_BUILD_DIR.exists() and (REACT_BUILD_DIR / "favicon.ico").exists():
        ico_path = REACT_BUILD_DIR / "favicon.ico"
    elif (BASE_DIR / "static" / "favicon.ico").exists():
        ico_path = BASE_DIR / "static" / "favicon.ico"

    if ico_path:
        return FileResponse(str(ico_path))

    # fall back to svg
    svg = FAV_DIST if FAV_DIST.exists() else (FAV_LEGACY if FAV_LEGACY.exists() else None)
    if svg:
        return FileResponse(str(svg), media_type="image/svg+xml")

    raise HTTPException(status_code=404, detail="favicon not found")


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

    # Consider the configured transcription service and readiness
    transcription_ready = True
    if TRANSCRIPTION_CONFIG.use_whisperx():
        transcription_ready = WHISPER_MODEL is not None

    status = "healthy" if (transcription_ready and LLM_HANDLER and volumes_ok) else "degraded"

    res = HealthResponse(
        status=status,
        # whisper_loaded strictly reflects local WhisperX model presence
        whisper_loaded=(WHISPER_MODEL is not None),
        llm_initialized=LLM_HANDLER is not None,
        supported_procedures=["col", "eus", "ercp", "egd"],
        transcription_service=TRANSCRIPTION_CONFIG.service,
        transcription_ready=transcription_ready
    )
    print("Health check:", res.json())
    return res


@app.get("/gpu-info")
async def gpu_info():
    """GPU diagnostics endpoint"""
    info = {
        "device": DEVICE,
        "whisperx_device": WHISPER_DEVICE,  # Actual device WhisperX is using
    }
    if TORCH_AVAILABLE:
        try:
            info.update({
                "pytorch_version": torch.__version__,
                "cuda_available": getattr(torch.cuda, 'is_available', lambda: False)(),
                "mps_available": hasattr(getattr(torch, 'backends', None), "mps") and torch.backends.mps.is_available(),
            })
        except Exception:
            info.update({"pytorch_version": None, "cuda_available": False, "mps_available": False})
    else:
        info.update({"pytorch_version": None, "cuda_available": False, "mps_available": False})

    if DEVICE == "cuda" and TORCH_AVAILABLE:
        try:
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                "gpu_count": torch.cuda.device_count(),
            })
        except Exception:
            # If querying GPU properties fails, return best-effort info
            pass
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
    1. Client sends JSON: {"type": "start", "session_id": "optional"}
    2. Client sends binary audio chunks (WebM, WAV, etc.)
    3. Server transcribes and sends back: {"type": "transcript", "data": {"text": "...", "session_id": "..."}}
    4. Client sends JSON: {"type": "end"} to finalize
    """
    client = getattr(websocket, 'client', None)
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted from: {client}")
    except Exception as e:
        print(f"Failed to accept WebSocket from {client}: {e}")
        raise
    session_id = None

    try:
        while True:
            try:
                data = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError) as e:
                # Uvicorn may RuntimeError when receive() called after a disconnect message has been received. Treat this as a normal disconnect
                print(f"WebSocket disconnected for session {session_id}: {e}")
                break
            except Exception as e:
                print(f"Error receiving WebSocket data from {client}: {e}")
                traceback.print_exc()
                break

            # Handle text (control messages)
            if "text" in data:
                try:
                    print(f"WebSocket text message from {client}: {str(data['text'])[:200]}")
                except Exception:
                    pass
                try:
                    last_activity_time = time.time()
                except Exception:
                    pass
                try:
                    message = json.loads(data["text"])
                    msg_type = message.get("type")

                    if msg_type == "start":
                        session_id = message.get("session_id") or str(uuid.uuid4())
                        # Accept opt client-reported chunk int so server calculate overlap in chunk counts correctly.
                        client_chunk_interval = message.get("chunk_interval_ms")
                        if client_chunk_interval is None:
                            client_chunk_interval = CONFIG.DEFAULT_CHUNK_INTERVAL_MS

                        SESSIONS[session_id] = {
                            "chunks": [],                    # all chunks
                            "transcripts": [],               # all transcript texts
                            "started_at": datetime.now(),
                            "buffer_chunks": [],             # chunks waiting to be transcribed
                            "buffer_start_time": None,
                            "last_transcribed_idx": 0,
                            "chunk_interval_ms": int(client_chunk_interval),
                        }

                        # If using Azure Speech streaming, initialize per-session PushAudioInputStream
                        if TRANSCRIPTION_CONFIG.use_azure():
                            try:
                                import azure.cognitiveservices.speech as speechsdk
                                # create push stream and recognizer
                                push_stream = speechsdk.audio.PushAudioInputStream()
                                audio_input = speechsdk.audio.AudioConfig(stream=push_stream)
                                speech_config = speechsdk.SpeechConfig(
                                    subscription=os.getenv("AZURE_SPEECH_KEY"),
                                    region=os.getenv("AZURE_SPEECH_REGION", "eastus")
                                )

                                recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

                                # capture event loop to schedule websocket sends from callback threads
                                loop = asyncio.get_event_loop()

                                def _recognized_cb(evt):
                                    try:
                                        result = evt.result
                                        text = getattr(result, 'text', '')
                                        if text:
                                            # append to session transcripts
                                            SESSIONS[session_id]["transcripts"].append(text)
                                            # send partial/final transcripts to client via event loop
                                            asyncio.run_coroutine_threadsafe(
                                                websocket.send_json({
                                                    "type": "transcript",
                                                    "data": {"text": text, "session_id": session_id}
                                                }),
                                                loop
                                            )
                                    except Exception as e:
                                        print(f"Azure recognized callback error: {e}")

                                def _canceled_cb(evt):
                                    try:
                                        print(f"Azure recognition canceled: {getattr(evt, 'cancellation_details', None)}")
                                    except Exception:
                                        pass

                                def _stopped_cb(evt):
                                    try:
                                        print(f"Azure recognition session stopped for {session_id}")
                                    except Exception:
                                        pass

                                recognizer.recognized.connect(_recognized_cb)
                                recognizer.canceled.connect(_canceled_cb)
                                recognizer.session_stopped.connect(_stopped_cb)

                                # start continuous recognition in background
                                try:
                                    recognizer.start_continuous_recognition()
                                except Exception as e:
                                    print(f"Failed to start Azure continuous recognition: {e}")

                                # store azure objects on session for later use
                                SESSIONS[session_id]["azure_push_stream"] = push_stream
                                SESSIONS[session_id]["azure_recognizer"] = recognizer
                                print(f"Initialized Azure streaming recognizer for session {session_id}")
                            except Exception as e:
                                print(f"Failed to initialize Azure streaming recognizer: {e}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Azure initialization failed: {e}",
                                    "session_id": session_id
                                })
                        await websocket.send_json({
                            "type": "status",
                            "message": "Session started",
                            "session_id": session_id
                        })
                        print(f"Started session {session_id}")

                    elif msg_type == "end":
                        # Close any streaming recognizer (Azure) and send the final transcript
                        if session_id and session_id in SESSIONS:
                            session = SESSIONS[session_id]
                            # If Azure streaming is active, close push stream and stop recognizer
                            if session.get("azure_push_stream") and session.get("azure_recognizer"):
                                try:
                                    print(f"Closing Azure push stream for {session_id}")
                                    session["azure_push_stream"].close()
                                except Exception as e:
                                    print(f"Error closing azure push stream: {e}")
                                try:
                                    loop = asyncio.get_event_loop()
                                    await loop.run_in_executor(None, session["azure_recognizer"].stop_continuous_recognition)
                                except Exception as e:
                                    print(f"Error stopping azure recognizer: {e}")

                            final_transcript = " ".join(session["transcripts"])
                            await websocket.send_json({
                                "type": "final",
                                "data": {
                                    "text": final_transcript,
                                    "session_id": session_id
                                },
                                "message": "Session ended"
                            })
                            print(f"Ended session {session_id}, sent final transcript ({len(final_transcript)} chars)")
                        else:
                            await websocket.send_json({
                                "type": "status",
                                "message": "Session ended",
                                "session_id": session_id
                            })
                        break

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })

            # Handle binary audio data
            elif "bytes" in data:
                try:
                    size = len(data.get("bytes") or b"")
                except Exception:
                    size = None
                print(f"WebSocket binary message from {client}: {size} bytes")
                # Update last-activity timestamp on websocket binary or text activity
                try:
                    last_activity_time = time.time()
                except Exception:
                    pass
                if not session_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No active session. Send 'start' message first."
                    })
                    continue

                if TRANSCRIPTION_CONFIG.use_whisperx() and WHISPER_MODEL is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Whisper model not initialized"
                    })
                    continue

                audio_data = data["bytes"]
                print(f"Received {len(audio_data)} bytes of audio for session {session_id}")

                session = SESSIONS[session_id]

                # REAL TIME: If Azure push stream exists for this session, convert the incoming blob to PCM16LE 16k and write directly into the PushAudioInputStream.
                if session.get("azure_push_stream"):
                    try:
                        loop = asyncio.get_event_loop()
                        pcm_bytes = await loop.run_in_executor(None, _convert_bytes_to_pcm16le_16k, audio_data)
                        try:
                            session["azure_push_stream"].write(pcm_bytes)
                        except Exception as e:
                            print(f"Failed to write to Azure push stream for {session_id}: {e}")
                        # Don't perform file-saving/buffering when streaming
                        continue
                    except Exception as e:
                        print(f"Azure streaming conversion/push failed: {e}")

                # Save audio chunk to temporary file
                chunk_id = str(uuid.uuid4())
                audio_path = UPLOAD_DIR / f"{session_id}_{chunk_id}.webm"

                with open(audio_path, "wb") as f:
                    f.write(audio_data)

                session["chunks"].append(str(audio_path))
                session["buffer_chunks"].append(str(audio_path))
                
                # Initialize buffer start time on first chunk
                if session["buffer_start_time"] is None:
                    session["buffer_start_time"] = time.time()

                buffer_duration = (time.time() - session["buffer_start_time"]) * 1000
                should_transcribe = buffer_duration >= CONFIG.TRANSCRIPTION_BUFFER_DURATION_MS
                
                print(f"Buffer: {len(session['buffer_chunks'])} chunks, {buffer_duration:.0f}ms (threshold: {CONFIG.TRANSCRIPTION_BUFFER_DURATION_MS}ms)")
                
                if should_transcribe and len(session["buffer_chunks"]) > 0:
                    try:
                        # Start a short-lived keepalive to prevent idle-shutdown during long-running transcription.
                        keepalive_task = None
                        async def _transcription_keepalive():
                            global last_activity_time
                            try:
                                while True:
                                    last_activity_time = time.time()
                                    await asyncio.sleep(min(5, max(1, CONFIG.IDLE_TIMEOUT_SECONDS // 2)))
                            except asyncio.CancelledError:
                                return

                        print(f"Transcribing {len(session['buffer_chunks'])} buffered chunks...")

                        # Send processing status to client
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Processing {len(session['buffer_chunks'])} chunks...",
                            "session_id": session_id
                        })

                        # If configured to use WhisperX, require the pre-loaded global model
                        if TRANSCRIPTION_CONFIG.use_whisperx() and WHISPER_MODEL is None:
                            raise Exception("WhisperX model not initialized")

                        # Concatenate buffered chunks for better transcription
                        concat_path = UPLOAD_DIR / f"{session_id}_buffer_{int(time.time())}.wav"
                        
                        # Define transcription function to run in thread
                        def transcribe_buffered_audio():
                            # Concatenate chunks into one file (like pep_risk does)
                            audio_file = concatenate_audio_chunks(
                                session["buffer_chunks"], 
                                str(concat_path)
                            )

                            if audio_file is None:
                                raise Exception("Failed to concatenate audio chunks")
                            return transcribe_audio(
                                audio_file,
                                service=TRANSCRIPTION_CONFIG.service,
                                device=DEVICE,
                                whisper_model="large-v3"
                            )

                        # Run transcription in thread pool to avoid blocking event loop
                        loop = asyncio.get_event_loop()
                        # if CONFIG.ENABLE_IDLE_SHUTDOWN:
                        #     keepalive_task = asyncio.create_task(_transcription_keepalive())
                        try:
                            result = await loop.run_in_executor(None, transcribe_buffered_audio)
                        finally:
                            if keepalive_task and not keepalive_task.done():
                                keepalive_task.cancel()
                        
                        transcript_text = result["text"]
                        session["transcripts"].append(transcript_text)

                        # Send transcription back to client
                        await websocket.send_json({
                            "type": "transcript",
                            "data": {
                                "text": transcript_text,
                                "session_id": session_id,
                                "chunk_count": len(session["buffer_chunks"]),
                                "timestamp": time.time()
                            }
                        })
                        print(f"Sent buffered transcript ({len(transcript_text)} chars): {transcript_text[:100]}...")

                        # Reset buffer for next transcription with overlap
                        chunk_interval_ms = session.get("chunk_interval_ms", CONFIG.DEFAULT_CHUNK_INTERVAL_MS)
                        try:
                            overlap_chunks = int(CONFIG.TRANSCRIPTION_BUFFER_OVERLAP_MS / max(1, int(chunk_interval_ms)))
                        except Exception:
                            overlap_chunks = 0

                        if overlap_chunks > 0 and len(session["buffer_chunks"]) > overlap_chunks:
                            session["buffer_chunks"] = session["buffer_chunks"][-overlap_chunks:]
                        else:
                            session["buffer_chunks"] = []
                        session["buffer_start_time"] = time.time()
                        
                        try:
                            os.remove(concat_path)
                        except:
                            pass

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
        if session_id and session_id in SESSIONS:
            print(f"Cleaning up session {session_id}")
            # delete temporary audio files
            for chunk_path in SESSIONS[session_id]["chunks"]:
                try:
                    os.remove(chunk_path)
                except:
                    pass


@app.post("/api/process", response_model=ProcessResponse)
async def process_transcript(request: ProcessRequest):
    """
    Process a transcript and extract structured data
    Args:
        request: ProcessRequest with transcript, procedure_type, and optional session_id
    Returns:
        ProcessResponse with extracted structured data
    """
    start_time = time.time()

    if LLM_HANDLER is None:
        raise HTTPException(status_code=503, detail="LLM handler not initialized")
    processor = PROCESSOR_MAP.get(request.procedure_type.value)
    if not processor:
        raise HTTPException(
            status_code=400,
            detail=f"Processor not available for procedure type: {request.procedure_type}"
        )

    try:
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
                    prompt_field_definitions_fp='./prompts/col/colonoscopies.txt',
                    fewshot_examples_dir="./prompts/col/fewshot",
                    prefix="col"
                )
                if LLM_HANDLER.model_type in ["openai", "anthropic"]:
                    col_response = LLM_HANDLER.chat(col_messages)
                else:
                    col_response = LLM_HANDLER.chat(col_messages)[0].outputs[0].text.strip()
                print(f"Col response: {col_response[:500]}")  
                start_idx = col_response.find("{")
                end_idx = col_response.rfind("}")
                if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                    raise ValueError(f"No valid JSON found in response. Response: {col_response[:500]}")
                json_str = col_response[start_idx:end_idx + 1]
                col_json = json.loads(json_str)

                try:
                    col_data_validated = ColonoscopyData(**col_json)
                    col_data = col_data_validated.model_dump()
                    col_data["participant_id"] = row["participant_id"]
                except ValidationError as e:
                    print(f"Colonoscopy validation errors: {e}")
                    # Fall back to basic validation
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
                start_idx = polyp_response.find("[")
                end_idx = polyp_response.rfind("]")
                if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                    raise ValueError(f"No valid JSON array found in polyp response. Response: {polyp_response[:500]}")
                json_str = polyp_response[start_idx:end_idx + 1]
                polyps_json = json.loads(json_str)
            
                validated_polyps = []
                for idx, polyp_json in enumerate(polyps_json):
                    try:
                        polyp_validated = PolypData(**polyp_json)
                        polyp_dict = polyp_validated.model_dump()
                        polyp_dict["participant_id"] = row["participant_id"]
                        polyp_dict["polyp_id"] = idx
                        validated_polyps.append(polyp_dict)
                    except ValidationError as e:
                        print(f"Polyp {idx+1} validation errors: {e}")
                        # Fall back to basic validation 
                        polyp_dict = polyp_json.copy()
                        polyp_dict["participant_id"] = row["participant_id"]
                        polyp_dict["polyp_id"] = idx
                        validated_polyps.append(polyp_dict)
                
                polyp_outputs.extend(validated_polyps)

            result_data = {
                "colonoscopy": col_outputs[0] if col_outputs else {},
                "polyps": polyp_outputs
            }

        elif request.procedure_type == ProcedureType.PEP_RISK:
            # PEP risk extraction and prediction #TODO PEP
            # 1. Extract risk factors
            # 2. Combine with manual inputs
            # 3. Feed to R model
            from pep_risk.peprisc_model import predict_pep_risk
            
            outputs = []

            for _, row in transcript_df.iterrows():
                llm_result = processor.extract_pep_from_transcript(
                    row["pred_transcript"],
                    filename=row["participant_id"]
                )
                outputs.append(llm_result)

            llm_extracted_data = outputs[0] if outputs else {}
            
            #?? manual data incorporate
            manual_data = None
            if request.manual_pep_data:
                manual_data = request.manual_pep_data.model_dump(exclude_none=True)
            # prediction model
            prediction_result = predict_pep_risk(
                manual_data=manual_data,
                llm_extracted_data=llm_extracted_data
            )
            
            result_data = {
                "llm_extracted": llm_extracted_data,
                "manual_input": manual_data,
                "prediction": prediction_result
            }
            # Add risk score, category and treatment predictions
            pep_risk_score = prediction_result.get("risk_score") if prediction_result.get("success") else None
            pep_risk_category = prediction_result.get("risk_category") if prediction_result.get("success") else None
            treatment_predictions = prediction_result.get("treatment_predictions", []) if prediction_result.get("success") else []

        else:
            # EUS, ERCP, EGD
            outputs = []
            for _, row in transcript_df.iterrows():
                # Determine the correct prompt field definitions file
                prompt_files = {
                    "ercp": "./prompts/ercp/generated_ercp_base_prompt.txt",
                    "col": "./prompts/col/generated_col_base_prompt.txt",
                    "egd": "./prompts/egd/generated_egd_base_prompt.txt",
                    "eus": "./prompts/eus/generated_eus_base_prompt.txt",
                }

                messages = processor.build_messages(
                    row["pred_transcript"],
                    prompt_field_definitions_fp=prompt_files.get(request.procedure_type.value, ""),
                    fewshot_examples_dir=f"./prompts/{request.procedure_type.value}/fewshot",
                    prefix=request.procedure_type.value
                )

                if LLM_HANDLER.model_type in ["openai", "anthropic"]:
                    response = LLM_HANDLER.chat(messages)
                else:
                    response = LLM_HANDLER.chat(messages)[0].outputs[0].text.strip()

                response_json = json.loads(response[response.find("{"): response.rfind("}") + 1])
                print("ERCP RESPONSE", response_json)
                
                # VALIDATE with Pydantic model
                procedure_models = {
                    "eus": EUSData,
                    "ercp": ErcpBaseData,  # Use generated model that matches the YAML/prompt
                    "egd": EGDData
                }
                model_class = procedure_models.get(request.procedure_type.value)
                if model_class:
                    try:
                        validated_data = model_class(**response_json)
                        result_dict = validated_data.model_dump()
                        result_dict["id"] = row["participant_id"]
                        result_dict["model"] = LLM_HANDLER.model_type
                        outputs.append(result_dict)
                    except ValidationError as e:
                        print(f"{request.procedure_type.value.upper()} validation errors: {e}")
                        # fall back to unvalidated
                        outputs.append({
                            "id": row["participant_id"],
                            "model": LLM_HANDLER.model_type,
                            **response_json
                        })
                else: # no model available, use unvalidated data
                    outputs.append({
                        "id": row["participant_id"],
                        "model": LLM_HANDLER.model_type,
                        **response_json
                    })
            result_data = outputs[0] if outputs else {}
            
            # If ERCP, also try PEP risk extraction+prediction
            pep_risk_score = None
            pep_risk_category = None
            pep_llm_extracted = None
            try:
                if request.procedure_type == ProcedureType.ERCP:
                    pep_processor = PROCESSOR_MAP.get("pep_risk")
                    if pep_processor:
                        # Use same transcript row (first row)
                        try:
                            first_transcript = transcript_df.iloc[0]["pred_transcript"]
                            first_id = transcript_df.iloc[0]["participant_id"]
                        except Exception:
                            first_transcript = request.transcript
                            first_id = request.session_id or "web_session"

                        pep_llm_extracted = pep_processor.extract_pep_from_transcript(
                            first_transcript,
                            filename=first_id
                        )
                        from pep_risk.peprisc_model import predict_pep_risk
                        manual_data = None
                        if request.manual_pep_data:
                            manual_data = request.manual_pep_data.model_dump(exclude_none=True)
                        prediction_result = predict_pep_risk(
                            manual_data=manual_data,
                            llm_extracted_data=pep_llm_extracted
                        )
                        if prediction_result.get("success"):
                            pep_risk_score = prediction_result.get("risk_score")
                            pep_risk_category = prediction_result.get("risk_category")
                            treatment_predictions = prediction_result.get("treatment_predictions", [])
            except Exception as e:
                print(f"PEP risk computation for ERCP failed: {e}")

        processing_time = time.time() - start_time
        
        ### Prep + populate procedure-specific fields for frontend
        response_data = ProcessResponse(
            success=True,
            procedure_type=request.procedure_type.value,
            session_id=request.session_id,
            data=result_data,
            processing_time_seconds=round(processing_time, 2)
        )
        if request.procedure_type == ProcedureType.COL:
            response_data.colonoscopy_data = result_data.get("colonoscopy")
            response_data.polyps_data = result_data.get("polyps")
        elif request.procedure_type == ProcedureType.PEP_RISK:
            response_data.pep_risk_data = result_data.get("llm_extracted")
            response_data.pep_risk_score = pep_risk_score if 'pep_risk_score' in locals() else None
            response_data.pep_risk_category = pep_risk_category if 'pep_risk_category' in locals() else None
            response_data.treatment_predictions = treatment_predictions if 'treatment_predictions' in locals() else []
        else:
            # EUS, ERCP, EGD
            response_data.procedure_data = result_data

        # If we computed PEP risk as part of ERCP processing above, include it on the response
        try:
            if request.procedure_type == ProcedureType.ERCP:
                response_data.pep_risk_data = pep_llm_extracted if 'pep_llm_extracted' in locals() and pep_llm_extracted else response_data.pep_risk_data
                response_data.pep_risk_score = pep_risk_score if 'pep_risk_score' in locals() else None
                response_data.pep_risk_category = pep_risk_category if 'pep_risk_category' in locals() else None
                response_data.treatment_predictions = treatment_predictions if 'treatment_predictions' in locals() else response_data.treatment_predictions
        except Exception:
            pass


        ### DRAFTER PLAIN TEXT with templating/drafter templates
        try:
            from templating.drafter_engine import build_report_sections
            proc = request.procedure_type.value
            cfg_fp = None
            candidate_fp = BASE_DIR.parent / 'drafters' / 'procedures' / proc / f'generated_{proc}_base.yaml'
            if candidate_fp.exists():
                cfg_fp = str(candidate_fp)
 
            if cfg_fp:
                # Prepare data for templates
                data_for_templates = response_data.procedure_data or response_data.data or result_data or {}
                sections = build_report_sections(cfg_fp, data_for_templates)
                # Join non-empty sections with headings
                parts = []
                for k in ['indications', 'history', 'description_of_procedure', 'findings', 'ercp_quality_metrics', 'impressions', 'recommendations']:
                    v = sections.get(k)
                    if v and v.strip():
                        parts.append(f"## {k.replace('_', ' ').title()}\n" + v.strip())
                rendered_note = "\n\n".join(parts).strip()
                if rendered_note:
                    response_data.formatted_note = rendered_note
        except Exception as e:
            print(f"Note rendering dailed: {e}")

        if request.session_id and request.session_id in SESSIONS:
            SESSIONS[request.session_id]["processed"] = True
            SESSIONS[request.session_id]["procedure_type"] = request.procedure_type.value
            SESSIONS[request.session_id]["results"] = response_data.model_dump()

            # Persist final outputs so History loads without re-running the LLM
            try:
                _persist_session_record(
                    request.session_id,
                    {
                        "procedure_type": request.procedure_type.value,
                        "created_at": SESSIONS[request.session_id].get("started_at", datetime.now()).isoformat(),
                        "processed": True,
                        "transcript": request.transcript,
                        "results": SESSIONS[request.session_id]["results"],
                    },
                )
            except Exception as e:
                print(f"Failed to persist session {request.session_id}: {e}")
        
        return response_data

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


@app.get("/api/sessions")
async def list_sessions():
    """List all active and completed sessions"""
    by_id: Dict[str, Dict[str, Any]] = {}

    # Persisted sessions first (survive restarts)
    for rec in _list_persisted_session_records():
        sid = rec.get("session_id")
        if not sid:
            continue
        by_id[sid] = {
            "session_id": sid,
            "procedure_type": rec.get("procedure_type", "unknown"),
            "created_at": rec.get("created_at") or rec.get("updated_at") or datetime.now().isoformat(),
            "processed": bool(rec.get("processed", False)),
            "transcript": rec.get("transcript", "") or "",
        }

    # In-memory sessions (live / current)
    for session_id, session_data in SESSIONS.items():
        # Do not overwrite persisted with less complete data.
        if session_id in by_id:
            continue
        by_id[session_id] = {
            "session_id": session_id,
            "procedure_type": session_data.get("procedure_type", "unknown"),
            "created_at": session_data.get("started_at", datetime.now()).isoformat(),
            "processed": session_data.get("processed", False),
            "transcript": " ".join(session_data.get("transcripts", [])),
        }

    sessions_list = list(by_id.values())
    sessions_list.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return sessions_list


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific session data"""
    rec = _load_persisted_session_record(session_id)
    if rec:
        return {
            "session_id": session_id,
            "procedure_type": rec.get("procedure_type", "unknown"),
            "created_at": rec.get("created_at") or rec.get("updated_at") or datetime.now().isoformat(),
            "processed": bool(rec.get("processed", False)),
            "transcript": rec.get("transcript", "") or "",
            "results": rec.get("results"),
        }

    session_data = SESSIONS.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "procedure_type": session_data.get("procedure_type", "unknown"),
        "created_at": session_data.get("started_at", datetime.now()).isoformat(),
        "processed": session_data.get("processed", False),
        "transcript": " ".join(session_data.get("transcripts", [])),
        "results": session_data.get("results"),
    }


@app.post("/api/sessions/{session_id}/save")
async def save_session(session_id: str, request: SaveSessionRequest):
    """Save/finalize the note content for a session without re-running the LLM."""
    # Prefer persisted record if it exists.
    rec = _load_persisted_session_record(session_id) or {}

    # Merge in-memory if present.
    mem = SESSIONS.get(session_id) or {}
    created_at = (
        rec.get("created_at")
        or rec.get("updated_at")
        or mem.get("started_at", datetime.now()).isoformat()
    )
    procedure_type = rec.get("procedure_type") or mem.get("procedure_type") or request.procedure_type or "unknown"
    transcript = rec.get("transcript") or (" ".join(mem.get("transcripts", [])) if mem else "") or request.transcript or ""

    # Results: prefer existing persisted, else in-memory, else provided.
    results = rec.get("results") or mem.get("results") or request.results
    if not isinstance(results, dict):
        results = {}

    # Save the final note into formatted_note.
    results["formatted_note"] = request.note_content

    record = {
        "procedure_type": procedure_type,
        "created_at": created_at,
        "processed": True,
        "transcript": transcript,
        "results": results,
    }
    _persist_session_record(session_id, record)

    # Keep in-memory session consistent when present.
    if session_id in SESSIONS:
        SESSIONS[session_id]["processed"] = True
        SESSIONS[session_id]["procedure_type"] = procedure_type
        SESSIONS[session_id]["results"] = results

    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    import sys
    
    print("=" * 50, flush=True)
    print("Starting EndoScribe server...", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"PORT env var: {os.getenv('PORT', 'not set')}", flush=True)
    print("=" * 50, flush=True)
    
    #! use `PORT` environment variable set by Fly `fly.toml` so app listens on the expected internal port, default 8000
    port = int(os.getenv("PORT", "8000"))
    print(f"Attempting to start server on 0.0.0.0:{port}", flush=True)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Failed to start server: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
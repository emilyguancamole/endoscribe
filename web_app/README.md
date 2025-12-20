# EndoScribe Web Application
**** NOTE 12/20/25 *** This README is pretty outdated and has been for a couple weeks. TODO update

A web-based interface for real-time medical transcription and structured data extraction from endoscopic procedures.

## Note Saving - 12/20
Where notes are saved:
- Local: saved to web_app/results/sessions/<session_id>.json (i.e, RESULTS_DIR set to BASE_DIR / "results").
- Fly: saved to /data/results/sessions/<session_id>.json (i.e, RESULTS_DIR set to /data/results). This persists across app restarts if Fly volume attached mounted at data.
A browser refresh will not remove the saved JSON — the saved note is persistent on disk


## System Requirements

### Required System Dependencies
- **Python 3.10+**
- **ffmpeg** (for audio processing) - **CRITICAL: Must be installed before running**
- **Homebrew** (macOS) or package manager for your OS

### Recommended Hardware
- **RAM**: 16GB+ (WhisperX model requires ~2GB)
- **Storage**: 10GB+ free space (for models)
- **CPU**: Multi-core processor (Apple Silicon M1/M2 or Intel i5+)
- **GPU**: Optional (CUDA for faster transcription)

## Quick Start

### 1. Install System Dependencies

**macOS:**
```bash
# Install ffmpeg (required for audio processing)
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install ffmpeg
```

### 2. Install Python Dependencies

```bash
# Install all Python packages from requirements.txt
pip install -r requirements.txt
```

**Note:** All required Python dependencies (FastAPI, Uvicorn, WebSockets, WhisperX, etc.) are in `requirements.txt`.

### 3. Configure Environment

```bash
# Set up environment variables (.env file)
echo "ANTHROPIC_API_KEY=your_api_key_here" >> .env
```

### 4. Start the Server

```bash
# From project root
./run_web_app.sh
# OR
python web_app/server.py
```

### 5. Open Browser

**http://localhost:8000**

## Features

- **Real-time Audio Recording**: Browser-based audio capture using MediaRecorder API
- **Live Transcription**: WebSocket-powered streaming transcription with WhisperX
- **Structured Data Extraction**: LLM-based extraction of procedure-specific data
- **Multi-Procedure Support**: Colonoscopy, EUS, ERCP, and EGD procedures
- **Modern UI**: Built with HTMX, Tailwind CSS, and DaisyUI

## Architecture

### Backend (FastAPI)
- **WebSocket endpoint** (`/ws/transcribe`): Handles real-time audio streaming and transcription
- **REST endpoint** (`/api/process`): Processes transcripts and extracts structured data
- **Health check** (`/health`): System status monitoring

### Frontend
- **Vanilla JavaScript**: MediaRecorder and WebSocket communication
- **HTMX**: Dynamic UI updates
- **Tailwind CSS + DaisyUI**: Styling and components

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file contains necessary API keys:

```bash
# For Anthropic Claude
ANTHROPIC_API_KEY=your_api_key_here

# For Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key_here

# For HuggingFace (for WhisperX)
HF_TOKEN=your_hf_token_here
```

### 3. Verify Prompts Directory Structure

The app expects prompt files in the following structure:

```
prompts/
├── col/
│   ├── system.txt
│   ├── colonoscopies.txt
│   ├── polyps.txt
│   └── fewshot/
├── eus/
│   ├── system.txt
│   ├── eus.txt
│   └── fewshot/
├── ercp/
│   ├── system.txt
│   ├── ercp.txt
│   └── fewshot/
└── egd/
    ├── system.txt
    ├── egd.txt
    └── fewshot/
```

## Running the Application

### Start the Server

```bash
# From the project root directory
python web_app/server.py
```

Or using uvicorn directly:

```bash
uvicorn web_app.server:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Web Interface

Open your browser and navigate to:

```
http://localhost:8000
```

## Usage

### 1. Select Procedure Type
Choose the procedure type from the dropdown (Colonoscopy, EUS, ERCP, or EGD).

### 2. Record Audio
1. Click **"Start Recording"** to begin capturing audio
2. Speak clearly into your microphone
3. Watch real-time transcription appear in the text area
4. Click **"Pause"** to temporarily stop (can resume later)
5. Click **"Stop"** when finished

### 3. Submit for Processing
1. After stopping, click **"Submit for Processing"**
2. Wait for the LLM to extract structured data
3. View results in formatted tables below

### 4. View Results

**Colonoscopy Results:**
- Colonoscopy-level data (indications, BBPS scores, extent, findings, etc.)
- Polyp-level data (size, location, resection method, NICE/Paris classification, etc.)

**Other Procedures:**
- Procedure-specific structured data based on the selected type

## WebSocket Protocol

The WebSocket endpoint (`/ws/transcribe`) uses the following message format:

**Client → Server:**
```json
// Start session
{"type": "start", "session_id": "optional_session_id"}

// Binary audio chunks (WebM format)
[Binary audio data]

// End session
{"type": "end"}
```

**Server → Client:**
```json
// Status updates
{"type": "status", "message": "...", "session_id": "..."}

// Transcription results
{
  "type": "transcript",
  "data": {
    "text": "transcribed text...",
    "session_id": "...",
    "chunk_id": "...",
    "timestamp": 1234567890.123
  }
}

// Errors
{"type": "error", "message": "error details..."}
```

## REST API

### Process Transcript

**Endpoint:** `POST /api/process`

**Request:**
```json
{
  "transcript": "Full transcript text...",
  "procedure_type": "col",
  "session_id": "optional_session_id"
}
```

**Response:**
```json
{
  "success": true,
  "procedure_type": "col",
  "session_id": "...",
  "data": {
    "colonoscopy": {...},
    "polyps": [...]
  },
  "processing_time_seconds": 12.34
}
```

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "whisper_loaded": true,
  "llm_initialized": true,
  "supported_procedures": ["col", "eus", "ercp", "egd"]
}
```

## File Structure

```
web_app/
├── server.py              # FastAPI application
├── models.py              # Pydantic schemas
├── templates/
│   └── index.html         # Main UI
├── static/
│   └── app.js            # Frontend JavaScript
├── uploads/               # Temporary audio files
├── results/               # Processing results
└── README.md             # This file
```

## Troubleshooting

### Microphone Access Denied
- Ensure your browser has permission to access the microphone
- Try using HTTPS (required for some browsers)
- Check browser console for specific errors

### WebSocket Connection Failed
- Verify the server is running
- Check firewall settings
- Ensure WebSocket support in your browser

### Transcription Not Working
- Check WhisperX model loaded successfully (see server logs)
- Verify audio chunks are being sent (check browser console)
- Ensure sufficient GPU/CPU resources

### Processing Failed
- Verify LLM API keys are set in `.env`
- Check prompt files exist in the correct directories
- Review server logs for specific errors

## Performance Notes

- **WhisperX**: Loads `large-v3` model on startup (requires ~5GB VRAM)
- **Audio Chunks**: Sent every 3 seconds for real-time transcription
- **Processing Time**: Varies based on transcript length and LLM model (typically 5-30 seconds)

## Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support (macOS 11+)
- **Mobile**: Limited support (microphone access varies by browser)

## Development

### Running in Development Mode

```bash
uvicorn web_app.server:app --reload --log-level debug
```

### Testing WebSocket Locally

You can test the WebSocket endpoint using tools like:
- Browser DevTools Console
- Postman
- websocat CLI tool

## Testing Workflow

### End-to-End Test

1. **Start the server**
   ```bash
   python web_app/server.py
   ```

2. **Open browser to `http://localhost:8000`**

3. **Check system health**
   - Look for green "System Ready" badge in top right
   - If degraded/error, check server logs for issues

4. **Select procedure type**
   - Choose "Colonoscopy" from dropdown

5. **Test recording**
   - Click "Start Recording"
   - Grant microphone permissions when prompted
   - Speak a sample colonoscopy report (or play pre-recorded audio)
   - Watch transcription appear in real-time
   - Test pause/resume functionality
   - Click "Stop" when finished

6. **Submit for processing**
   - Click "Submit for Processing"
   - Wait for LLM extraction (5-30 seconds)
   - Verify results display correctly

7. **Verify results**
   - Check colonoscopy data table
   - Check polyps table (if polyps mentioned)
   - Validate data accuracy

### Sample Test Transcript

Use this sample for testing colonoscopy extraction:

```
This is a colonoscopy for a 65-year-old female with a personal history of colon polyps.
Her last colonoscopy was 1 year ago. Beginning the procedure now. Inserting the scope
through the anus. The Boston Bowel Prep Score is 3 in the right colon, 3 in the transverse,
and 3 in the left colon for a total of 9. The prep quality is excellent. We've reached the
cecum. The appendiceal orifice and ileocecal valve are visualized. I see two polyps in the
ascending colon measuring 5mm and 8mm. Both are Paris 0-Is, NICE type 2. Resecting with
cold snare. Polyps retrieved. Withdrawing the scope. No complications. Procedure complete.
```

## Important Notes

### Required Setup

1. **Environment Variables**
   - `ANTHROPIC_API_KEY` (server uses `anthropic_claude` config)
   - Optional: `AZURE_OPENAI_API_KEY`
   - Optional: `HF_TOKEN` for HuggingFace models

2. **Model Downloads**
   - WhisperX `large-v3` model downloads automatically on first startup (~5GB)
   - Requires sufficient disk space and good internet connection
   - First startup may take 5-10 minutes

3. **System Requirements**
   - **GPU recommended**: 8GB+ VRAM for WhisperX
   - **CPU mode**: Works but slower (transcription may lag)
   - **RAM**: 16GB+ recommended
   - **Disk**: 10GB+ free space

4. **Browser Requirements**
   - Modern browser with MediaRecorder API support
   - Microphone permissions required
   - HTTPS required for remote access (localhost works on HTTP)

### Audio Format

- **Recording Format**: WebM (browser native)
- **Chunk Interval**: 3 seconds (configurable in `app.js`)
- **Supported Formats**: WhisperX accepts WAV, MP3, M4A, WebM
- **Sample Rate**: Browser default (typically 48kHz)

### File Locations

- **Prompts**: `prompts/{procedure_type}/` (uses existing project structure)
- **Results**: `web_app/results/` (CSV outputs from processing)
- **Audio Uploads**: `web_app/uploads/` (temporary WebM chunks)
- **Logs**: Console output (add logging middleware if needed)
- **Static Assets**: `web_app/static/` (JavaScript files)
- **Templates**: `web_app/templates/` (HTML files)

## Troubleshooting - Detailed Solutions

### Issue: Server Won't Start

**Symptoms:**
- Import errors
- Module not found errors

**Solutions:**
```bash
# Ensure you're in the project root
cd /path/to/endoscribe

# Install all dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print(sys.path)"

# Try running with explicit path
PYTHONPATH=. python web_app/server.py
```

### Issue: Microphone Access Denied

**Symptoms:**
- Recording doesn't start
- Browser shows blocked icon

**Solutions:**
1. **Chrome/Edge**: Click lock icon in address bar → Site settings → Allow microphone
2. **Firefox**: Click permissions icon → Allow microphone
3. **Safari**: System Preferences → Security & Privacy → Microphone → Allow browser
4. **Remote access**: Must use HTTPS (localhost exempt)

### Issue: WebSocket Connection Failed

**Symptoms:**
- "Connection error occurred" alert
- Transcription doesn't appear

**Solutions:**
```bash
# Check server is running
curl http://localhost:8000/health

# Check WebSocket in browser console
# Should see: "WebSocket connected"

# Verify no proxy/firewall blocking WebSocket
# Check browser dev tools Network tab for WS connection

# Try different port if 8000 is blocked
uvicorn web_app.server:app --port 8001
```

### Issue: Transcription Not Working

**Symptoms:**
- Audio recorded but no transcription
- "Transcription error" in logs

**Solutions:**
1. **Check WhisperX loaded**: Look for "WhisperX model loaded successfully" in server logs
2. **GPU issues**: If CUDA error, try forcing CPU mode:
   ```python
   # In server.py, change line:
   DEVICE = "cpu"  # Force CPU mode
   ```
3. **Audio format issues**: Verify WebM chunks are valid
4. **Check server logs**: Look for specific WhisperX errors

### Issue: Processing Failed / LLM Errors

**Symptoms:**
- "Processing failed" error
- Long wait times with no result

**Solutions:**
1. **Verify API key**:
   ```bash
   # Check .env file
   cat .env | grep ANTHROPIC_API_KEY

   # Test API key
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $ANTHROPIC_API_KEY"
   ```

2. **Check prompt files exist**:
   ```bash
   ls -la prompts/col/system.txt
   ls -la prompts/col/colonoscopies.txt
   ```

3. **Review server logs** for specific error messages

4. **API rate limits**: Wait and retry if hitting rate limits

### Issue: Results Not Displaying

**Symptoms:**
- Processing completes but no results shown
- Blank results section

**Solutions:**
1. **Check browser console** for JavaScript errors
2. **Verify response format**: Should match Pydantic schemas
3. **Check procedure type**: Ensure correct processor was used
4. **Inspect network tab**: Verify `/api/process` response contains data

### Issue: WhisperX Model Won't Download

**Symptoms:**
- "Failed to load WhisperX model" error
- Timeout during download

**Solutions:**
```bash
# Download model manually
python -c "import whisperx; whisperx.load_model('large-v3', 'cpu')"

# Check HuggingFace access
# May need HF_TOKEN for some models

# Use smaller model if disk/bandwidth limited
# Edit server.py: whisperx.load_model("base", DEVICE)
```

## Customization Options

### 1. Change LLM Model

**File:** `web_app/server.py` (line 34)

```python
# Current: Anthropic Claude
LLM_HANDLER = LLMClient.from_config("anthropic_claude")

# Switch to OpenAI GPT-4
LLM_HANDLER = LLMClient.from_config("openai_gpt4o")

# Or use local model
LLM_HANDLER = LLMClient.from_config("local_llama")
```

### 2. Adjust Audio Chunk Interval

**File:** `web_app/static/app.js` (line 162)

```javascript
// Current: 3 second chunks
state.mediaRecorder.start(3000);

// Faster transcription (1 second chunks)
state.mediaRecorder.start(1000);

// Slower, more efficient (5 second chunks)
state.mediaRecorder.start(5000);
```

### 3. Change WhisperX

**File:** `web_app/server.py` (line 49)

```python
# large-v3 (~5GB)
WHISPER_MODEL = whisperx.load_model("large-v3", DEVICE, compute_type="float16")

### 5. Add User Authentication

**File:** `web_app/server.py`

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Add your auth logic here
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Protect endpoints
@app.post("/api/process", dependencies=[Depends(verify_token)])
async def process_transcript(request: ProcessRequest):
    # ... existing code
```

### 6. Enable HTTPS

**Option A: Uvicorn with SSL**
```bash
uvicorn web_app.server:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

### 7. Add Logging

**File:** `web_app/server.py` (add at top)

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Use in code
logger.info(f"Processing transcript for session {session_id}")
```

8. Auto-Delete Temporary Audio Files

## Production Deployment

### Docker Deployment (Optional)

Create `web_app/Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "web_app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run with Docker:
```bash
docker build -t endoscribe-web .
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env endoscribe-web
```

### Environment-Specific Configs

Consider using different configs for dev/staging/prod:

```python
import os

ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    # Production settings
    WHISPER_MODEL_SIZE = "large-v3"
    DEBUG = False
else:
    # Development settings
    WHISPER_MODEL_SIZE = "base"
    DEBUG = True
```

## License
Part of the EndoScribe project.

# Azure Speech Service Migration Guide

This guide walks you through migrating from WhisperX (GPU-based) to Azure Speech Service (managed cloud service).

## Overview

Your endoscribe project has been set up to support both WhisperX and Azure Speech Service transcription. You can switch between them easily or run them side-by-side during the migration period.

## Quick Start

### 1. Install Dependencies

```bash
cd /home/eguan2/endoscribe
pip install azure-cognitiveservices-speech
```

### 2. Configure Azure Credentials

Create or update your `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Azure credentials
nano .env
```

Add these lines to `.env`:

```env
# Azure Speech Service Configuration
AZURE_SPEECH_KEY=your_actual_api_key_here
AZURE_SPEECH_REGION=eastus
AZURE_SPEECH_ENDPOINT=https://eastus.api.cognitive.microsoft.com/

# Set default transcription service (optional)
TRANSCRIPTION_SERVICE=azure
```

### 3. Test Your Setup

Run the test script to verify Azure is working:

```bash
python transcription/test_azure.py path/to/test_audio.wav
```

Or let it auto-detect a test file:

```bash
python transcription/test_azure.py
```

## Usage Examples

### Option 1: Use the Unified Interface (Recommended)

The unified interface automatically selects Azure or WhisperX based on your configuration:

```python
from transcription.transcription_service import transcribe_unified

# Uses service specified in TRANSCRIPTION_SERVICE env var (or Azure by default)
result = transcribe_unified("audio.wav")

# Force specific service
result = transcribe_unified("audio.wav", service="azure")
result = transcribe_unified("audio.wav", service="whisperx")

# With speaker diarization (Azure only)
result = transcribe_unified(
    "audio.wav", 
    service="azure",
    enable_diarization=True,
    max_speakers=2
)

# Result format (consistent across both services)
{
    "text": "full transcript here",
    "segments": [
        {"text": "...", "start": 0.0, "end": 2.5},
        ...
    ],
    "service": "azure",  # or "whisperx"
    "language": "en-US",
    "duration": 120.5
}
```

### Option 2: Use Azure Directly

If you only want to use Azure:

```python
from transcription.azure_transcribe import transcribe_azure

result = transcribe_azure(
    audio_file="audio.wav",
    language="en-US",  # Language code
    enable_word_level_timestamps=True,
    enable_diarization=False
)
```

### Option 3: Keep Using WhisperX

Your existing code continues to work:

```python
from transcription.whisperx_transcribe import transcribe_whisperx

result = transcribe_whisperx("audio.wav", whisper_model="large-v3")
```

## Migrating Your Code

### In `pep_risk/server.py`

**Current code:**
```python
from transcription.whisperx_transcribe import transcribe_whisperx

result = transcribe_whisperx(audio_fp, whisper_model="large-v3", device=device)
```

**Option A - Minimal change (keeps both options):**
```python
from transcription.transcription_service import transcribe_unified

# Will use Azure if credentials are set, otherwise WhisperX
result = transcribe_unified(audio_fp, whisper_model="large-v3")
```

**Option B - Force Azure:**
```python
from transcription.azure_transcribe import transcribe_azure

result = transcribe_azure(audio_fp, language="en-US")
```

### In `web_app/server.py`

**Current code:**
```python
from transcription.whisperx_transcribe import transcribe_whisperx

result = transcribe_whisperx(audio_file, whisper_model="large-v3", device=device)
```

**Updated code:**
```python
from transcription.transcription_service import transcribe_unified

# Automatically uses configured service
result = transcribe_unified(audio_file)

# Or specify explicitly
result = transcribe_unified(audio_file, service="azure")
```

## Environment Variables

Add these to your `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_SPEECH_KEY` | Yes | - | Your Azure Speech API key |
| `AZURE_SPEECH_REGION` | No | `eastus` | Azure region (eastus, westus2, etc.) |
| `AZURE_SPEECH_ENDPOINT` | No | Auto-generated | Full endpoint URL |
| `TRANSCRIPTION_SERVICE` | No | `azure` | Default service: `azure` or `whisperx` |

## Comparing Azure vs WhisperX

### Azure Speech Service Advantages
- ✅ **No GPU required** - runs entirely in the cloud
- ✅ **Faster** - no model loading time, parallel processing
- ✅ **Managed service** - no infrastructure maintenance
- ✅ **Better diarization** - built-in speaker identification
- ✅ **Cost-effective** - pay per use, no GPU hardware costs
- ✅ **Scalable** - handles concurrent requests easily

### WhisperX Advantages
- ✅ **No API costs** - free after GPU infrastructure
- ✅ **Privacy** - data stays on your server
- ✅ **Offline capable** - works without internet
- ✅ **Fine-tuning** - can customize models for medical terms

### Performance Comparison

Run the test script with comparison enabled:

```bash
python transcription/test_azure.py audio.wav
# When prompted, type 'y' to compare with WhisperX
```

## Pricing

Azure Speech Service pricing (as of Dec 2024):
- **Standard**: $1 per audio hour
- **Free tier**: 5 audio hours/month free

For typical procedure recordings (10-30 minutes), costs are minimal ($0.17-$0.50 per procedure).

## Troubleshooting

### "AZURE_SPEECH_KEY not found"
- Make sure your `.env` file is in the project root: `/home/eguan2/endoscribe/.env`
- Verify the key is correct (no extra spaces or quotes)
- Restart your Python environment after updating `.env`

### "Authentication failed"
- Check that your Azure Speech Service is deployed in the correct region
- Verify your API key is active in Azure Portal
- Ensure your Azure subscription is active

### "Audio format not supported"
- Azure supports: WAV, MP3, OGG, FLAC, OPUS, WEBM
- Convert if needed: `ffmpeg -i input.m4a -ar 16000 output.wav`

### Poor transcription quality
- Ensure audio is clear (use noise reduction if needed)
- Try different language codes: `en-US` vs `en-GB`
- For medical terms, WhisperX with fine-tuning may be better

### Import errors
- Make sure you've installed the Azure SDK: `pip install azure-cognitiveservices-speech`
- Check that you're in the correct Python environment

## Gradual Migration Strategy

You can migrate gradually:

1. **Week 1**: Install Azure, test with a few files
2. **Week 2**: Run parallel (Azure + WhisperX) to compare quality
3. **Week 3**: Switch default to Azure, keep WhisperX as fallback
4. **Week 4**: Full Azure, decommission WhisperX GPU setup

Set this up with environment variables:

```python
# In your .env, control which service is used
TRANSCRIPTION_SERVICE=azure  # or "whisperx"
```

## API Reference

### transcribe_unified()

Main function for all transcription:

```python
def transcribe_unified(
    audio_file: str,           # Path to audio file
    service: Optional[str] = None,  # "azure" or "whisperx"
    whisper_model: str = "large-v3",  # WhisperX model
    device: Optional[str] = None,     # WhisperX device
    language: str = "en-US",          # Language code
    enable_diarization: bool = False, # Speaker ID
    **kwargs
) -> Dict
```

### transcribe_azure()

Azure-specific function:

```python
def transcribe_azure(
    audio_file: str,                        # Path to audio
    language: str = "en-US",                # Language code
    enable_word_level_timestamps: bool = True,
    enable_diarization: bool = False,       # Speaker ID
    max_speakers: int = 2
) -> Dict
```

## Next Steps

1. ✅ Install Azure SDK: `pip install azure-cognitiveservices-speech`
2. ✅ Configure `.env` with your Azure credentials
3. ✅ Run test: `python transcription/test_azure.py`
4. ⏭️ Update one module (e.g., `pep_risk/server.py`) to use `transcribe_unified()`
5. ⏭️ Test the updated module thoroughly
6. ⏭️ Gradually update other modules
7. ⏭️ Compare quality and costs over a trial period
8. ⏭️ Make final decision on default service

## Getting Your Azure API Key

If you don't have your API key handy:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Speech Service resource
3. Click "Keys and Endpoint" in the left menu
4. Copy "KEY 1" or "KEY 2"
5. Note the "Region" (e.g., eastus)

## Support

For issues:
- Check the troubleshooting section above
- Review Azure Speech Service [documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)
- Test with the provided test script: `python transcription/test_azure.py`

---

**Files Created:**
- `transcription/azure_transcribe.py` - Azure implementation
- `transcription/transcription_service.py` - Unified interface
- `transcription/test_azure.py` - Testing script
- `.env.example` - Environment template
- `AZURE_MIGRATION_GUIDE.md` - This guide

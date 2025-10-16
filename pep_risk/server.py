from fastapi import FastAPI, UploadFile, Form
import os, uuid, asyncio
from whisperx import load_model
import torch

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load WhisperX once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("large-v3", device=device)

@app.post("/upload")
async def upload_audio(file: UploadFile):
    """Receive audio chunk and transcribe."""
    # Save file
    file_id = str(uuid.uuid4())
    audio_fp = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    with open(audio_fp, "wb") as f:
        f.write(await file.read())

    # Transcribe with WhisperX
    print(f"Transcribing {audio_fp} ...")
    result = model.transcribe(audio_fp, batch_size=8, language="en")
    print("Transcription complete, result", result)
    text = result["segments"]["text"] # {'segments': [{'text': ' You will receive an email with a link to my hub.', 'start': 6.393, 'end': 10.308}], 'language': 'en'}

    print(f"Transcription: {text}")

    # todo alignment. Align whisper output to improve word level timing alignment... how does this work with chunks?
    # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # ---- Placeholder: replace with your LLM + PEP pipeline ----
    pep_risk = await process_pipeline(text)

    return {"transcript": text, "pep_risk": pep_risk}


async def process_pipeline(transcript: str):
    """
    Simulate sending transcript to LLM extraction and PEP risk model.
    Replace this with your actual pipeline.
    """
    await asyncio.sleep(1)
    # Example placeholder: simple rule
    if "pancreatic duct" in transcript.lower():
        return "High"
    elif "cannulation" in transcript.lower():
        return "Moderate"
    else:
        return "Low"

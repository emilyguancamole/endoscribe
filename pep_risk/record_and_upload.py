import sounddevice as sd
import soundfile as sf
import threading
import queue
import requests
import subprocess
import time
from datetime import datetime
import os
import uuid

SERVER_URL = "http://localhost:8000/upload" 
FINALIZE_URL = "http://localhost:8000/finalize"
AUDIO_PATH = "recording_buffer.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
CHUNK_DIR = "chunks"

os.makedirs(CHUNK_DIR, exist_ok=True)
audio_q = queue.Queue()

def record_audio():
    """Continuously record audio to rolling buffer."""
    with sf.SoundFile(AUDIO_PATH, mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS) as f:
        def callback(indata, frames, time, status):
            if status:
                print(status)
            f.write(indata)
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            print("Recording... Press Ctrl+C to stop.")
            threading.Event().wait()  # block forever


def extract_chunk(start_time, duration, output_fp):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", AUDIO_PATH,
        "-ac", "1", "-ar", "16000",
        output_fp
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


session_id = None  # global session for the current recording


def upload_chunk(fp, is_last=False):
    """Upload audio chunk to server."""
    global session_id
    if not os.path.exists(fp) or os.path.getsize(fp) < 10_000:  # ~10 KB threshold
        print(f"Skipping {fp}, file too small or missing.")
        return
    print(f"Uploading {fp}...")
    try:
        form = {"is_last": str(is_last).lower()}
        if session_id:
            form["session_id"] = session_id
        with open(fp, "rb") as f:
            resp = requests.post(SERVER_URL, files={"file": f}, data=form, timeout=600)  # 10 minutes for finalization
        if resp.ok:
            data = resp.json()
            if not session_id and data.get("session_id"):
                session_id = data["session_id"]
            print(f"PEP Risk: {data.get('pep_risk')} — transcript length: {len(data.get('transcript', ''))}")
        else:
            print("Upload failed", resp.text)
    except Exception as e:
        print(f"Upload error: {e}")

def chunk_loop():
    interval = 30
    overlap = 10
    start = 0
    print("Waiting for first buffer to fill...")
    time.sleep(interval)

    while True:
        output_fp = os.path.join(CHUNK_DIR, f"chunk_{datetime.now():%Y%m%d_%H%M%S}.wav")
        ss = max(start - overlap, 0)
        extract_chunk(ss, interval + overlap, output_fp)
        upload_chunk(output_fp, is_last=False)
        start += interval
        time.sleep(interval)


if __name__ == "__main__":
    """python record_and_upload.py
    Continuously record audio, chunk it, and upload to server.
    """
    recorder_thread = threading.Thread(target=record_audio, daemon=True)
    chunk_thread = threading.Thread(target=chunk_loop, daemon=True)

    recorder_thread.start()
    chunk_thread.start()

    try:
        recorder_thread.join()
    except KeyboardInterrupt:
        # On Ctrl+C, extract and upload the final chunk, then finalize
        print("\n\nStopping recording and finalizing session...")
        try:
            if session_id:
                # Extract the final chunk from the current buffer position
                import time
                current_time = time.time()
                # Get the last chunk from wherever we left off
                output_fp = os.path.join(CHUNK_DIR, f"chunk_final_{datetime.now():%Y%m%d_%H%M%S}.wav")
                
                # Use ffmpeg to extract from the buffer (get the last 30 seconds or whatever is there)
                # This ensures we capture any audio recorded after the last scheduled chunk
                cmd = [
                    "ffmpeg", "-y",
                    "-sseof", "-30",  # Start 30 seconds from end
                    "-i", AUDIO_PATH,
                    "-ac", "1", "-ar", "16000",
                    output_fp
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Upload this final chunk with is_last=true
                print("Uploading final chunk...")
                upload_chunk(output_fp, is_last=True)
                print("Finalization complete!")
        except Exception as e:
            print("Finalize error:", e)

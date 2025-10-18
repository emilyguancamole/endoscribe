import argparse
import os
import requests

SERVER_URL = "http://localhost:8000/upload"


def main():
    """Upload a pre-recorded audio file (wav, m4a, mp3) to the ERCP pipeline server.
    Sends is_last=true so the server transcribes once, concatenates (just the 1 transcript), runs ERCP extraction, and saves to CSV
    """
    parser = argparse.ArgumentParser(description="Upload a pre-recorded audio file to the ERCP pipeline server.")
    parser.add_argument("audio_path", help="Path to the audio file (wav, m4a, mp3)")
    parser.add_argument("--session-id", dest="session_id", default=None, help="Optional session id to reuse; otherwise a new one is created")
    parser.add_argument("--server", default=SERVER_URL, help="Server upload URL (default: http://localhost:8000/upload)")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    # Basic content-type guess (server doesn't strictly need it)
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in [".wav", ".m4a"]:
        print(f"Warning: extension {ext} is not a typical wav/m4a; attempting upload anyway.")

    form = {"is_last": "true"}
    if args.session_id:
        form["session_id"] = args.session_id

    print(f"Uploading {audio_path} as final chunk to {args.server} ...")
    with open(audio_path, "rb") as f:
        resp = requests.post(args.server, files={"file": f}, data=form, timeout=600)

    if not resp.ok:
        print("Upload failed:", resp.status_code, resp.text)
        return

    data = resp.json()
    print("--- Server Response ---")
    for k, v in data.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

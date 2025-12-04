#!/usr/bin/env python3
"""
Batch processor for PEPRISC uploads.

Posts filenames found in a local recordings directory to the running server endpoint `/process_local` and optionally runs multiple uploads in parallel, with retries and skip-processed support.

Usage examples:
  python3 scripts/process_peprisc_batch.py --dir pep_risk/recordings/peprisc_uploads_rec
  python3 scripts/process_peprisc_batch.py --dir pep_risk/recordings/peprisc_uploads_rec --concurrency 4 --skip-processed

The script expects the server (uvicorn pep_risk.server:app) to be running and the
`filename` argument corresponds to the basename inside the server's recordings folder.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from typing import List

try:
    import requests
except Exception:
    print("The 'requests' library is required. Install with: pip install requests")
    raise

import pandas as pd

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8002


def list_files(path: str) -> List[str]:
    exts = {".mp3", ".wav", ".m4a"}
    files = [f for f in sorted(os.listdir(path)) if os.path.splitext(f)[1].lower() in exts]
    return files


def load_processed_set(batch_csv_path: str):
    if not os.path.exists(batch_csv_path):
        return set()
    try:
        df = pd.read_csv(batch_csv_path)
        if "filename" in df.columns:
            return set(df["filename"].astype(str).tolist())
    except Exception:
        pass
    return set()


def process_file(host, port, filename, recordings_dir, retries=2, timeout=300):
    url = f"http://{host}:{port}/process_local"
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, data={"filename": filename}, timeout=timeout)
            try:
                data = resp.json()
            except Exception:
                data = {"status_code": resp.status_code, "text": resp.text}
            # If the server returned an application-level error inside JSON,
            # treat this as a failed processing attempt so callers can retry/inspect.
            if isinstance(data, dict):
                if data.get("error") or data.get("success") is False:
                    return False, data
            # Non-JSON responses with non-200 status are failures
            if not isinstance(data, dict) and resp.status_code != 200:
                return False, data

            return True, data
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return False, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", default="pep_risk/recordings/peprisc_uploads_rec")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--concurrency", "-c", type=int, default=2)
    parser.add_argument("--skip-processed", action="store_true")
    parser.add_argument("--batch-csv", default="pep_risk/results/batch_pep_predictions.csv")
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    recordings_dir = args.dir
    if not os.path.isdir(recordings_dir):
        print(f"Recordings directory not found: {recordings_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    files = list_files(recordings_dir)
    if not files:
        print("No audio files found.")
        return

    processed = set()
    if args.skip_processed:
        processed = load_processed_set(args.batch_csv)
        if processed:
            print(f"Skipping {len(processed)} already-processed files found in {args.batch_csv}")

    to_process = [f for f in files if (not args.skip_processed) or (f not in processed)]
    print(f"Found {len(files)} files, will process {len(to_process)}")

    lock = threading.Lock()
    results = []

    def worker(fname):
        ok, data = process_file(args.host, args.port, fname, recordings_dir, retries=args.retries)
        out_fp = os.path.join(args.out_dir, f"process_{fname}.json")
        try:
            with open(out_fp, "w") as fh:
                json.dump({"filename": fname, "ok": ok, "result": data}, fh)
        except Exception:
            pass
        with lock:
            results.append((fname, ok, data))
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {fname}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, f) for f in to_process]
        for fut in concurrent.futures.as_completed(futures):
            pass

    fails = [r for r in results if not r[1]]
    print(f"Batch finished. {len(results)} attempted, {len(fails)} failures.")
    if fails:
        print("Failed files:")
        for fname, ok, data in fails:
            print(" -", fname, data.get("error") if isinstance(data, dict) else str(data))


if __name__ == "__main__":
    main()

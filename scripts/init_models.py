#!/usr/bin/env python3
"""
Initialize WhisperX models for EndoScribe deployment.

This script pre-downloads WhisperX models to the persistent volume,
which speeds up subsequent application startups on Fly.io.

Usage:
    # On Fly.io (via SSH)
    python scripts/init_models.py

    # Locally
    python scripts/init_models.py --model-dir ./models
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
import whisperx


def download_whisperx_models(model_dir: Path, device: str = "cuda"):
    """
    Download WhisperX models to the specified directory.

    Args:
        model_dir: Directory to store models
        device: Device to use (cuda/cpu)
    """
    print(f"Downloading WhisperX models to {model_dir}")
    print(f"Device: {device}")

    # Ensure directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine compute type
    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"

    print(f"\n1. Loading WhisperX large-v3 model...")
    print(f"   Compute type: {compute_type}")
    try:
        model = whisperx.load_model(
            "large-v3",
            device=device,
            compute_type=compute_type,
            download_root=str(model_dir)
        )
        print("   ✓ WhisperX large-v3 model downloaded successfully!")
    except Exception as e:
        print(f"   ✗ Failed to download WhisperX model: {e}")
        return False

    print(f"\n2. Loading WhisperX alignment model...")
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code="en",
            device=device
        )
        print("   ✓ WhisperX alignment model downloaded successfully!")
    except Exception as e:
        print(f"   ✗ Failed to download alignment model: {e}")
        return False

    print(f"\n3. Model Summary:")
    print(f"   WhisperX large-v3: {compute_type}")
    print(f"   Alignment model: English")
    print(f"   Storage location: {model_dir}")

    # Check disk usage
    total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)
    print(f"   Total size: {size_gb:.2f} GB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Initialize WhisperX models for EndoScribe"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/data/models/whisperx"),
        help="Directory to store models (default: /data/models/whisperx)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        device = "cuda" if (TORCH_AVAILABLE and getattr(torch, 'cuda', None) and torch.cuda.is_available()) else "cpu"
        print(f"Auto-detected device: {device}")
    else:
        device = args.device

    # Verify CUDA if requested
    if device == "cuda" and not (TORCH_AVAILABLE and getattr(torch, 'cuda', None) and torch.cuda.is_available()):
        print("ERROR: CUDA requested but not available!")
        print("Falling back to CPU...")
        device = "cpu"

    # Show GPU info if available
    if device == "cuda" and TORCH_AVAILABLE:
        try:
            print(f"\nGPU Information:")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception:
            print("GPU info unavailable")

    # Set environment variables for model caching
    os.environ["HF_HOME"] = str(args.model_dir.parent / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(args.model_dir.parent / "transformers")
    os.environ["WHISPERX_MODELS"] = str(args.model_dir)

    print(f"\nStarting model download...")
    print(f"This may take 5-10 minutes depending on your connection.\n")

    success = download_whisperx_models(args.model_dir, device)

    if success:
        print("\n" + "="*60)
        print("SUCCESS! All models downloaded successfully.")
        print("="*60)
        print("\nYou can now start the EndoScribe server.")
        print("Models will be loaded from the cache on startup.")
        return 0
    else:
        print("\n" + "="*60)
        print("FAILED! Some models could not be downloaded.")
        print("="*60)
        print("\nPlease check the errors above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

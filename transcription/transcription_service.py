"""
Unified transcription interface.
Provides a simple way to switch between WhisperX (GPU) and Azure Speech Service.
"""
import os
from typing import Dict, Optional, List
import subprocess
from dotenv import load_dotenv

load_dotenv()

class TranscriptionConfig:
    """Configuration for transcription service selection."""
    WHISPERX = "whisperx"
    AZURE = "azure"
    def __init__(self):
        # Default to Azure if credentials are available, otherwise WhisperX
        self.service = os.getenv("TRANSCRIPTION_SERVICE", "azure")
        self.azure_key = os.getenv("AZURE_SPEECH_KEY")
        
        # Auto-detect if Azure available
        if self.service == "azure" and not self.azure_key:
            print("Warning: AZURE_SPEECH_KEY not found. Falling back to WhisperX.")
            self.service = self.WHISPERX
    def use_azure(self) -> bool:
        # Require a truthy/ non-empty AZURE key to consider Azure usable
        return self.service == self.AZURE and bool(self.azure_key)
    def use_whisperx(self) -> bool:
        return self.service == self.WHISPERX

def convert_to_wav(src, dest):
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dest])

def transcribe_unified(
    audio_files: Optional[List[str]] = None,
    service: Optional[str] = None,
    whisper_model: str = "large-v3",
    device: Optional[str] = None,
    enable_diarization: bool = False,
    procedure_type: Optional[str] = None,
    phrase_list: Optional[list] = None,
    save: bool = True,
    save_filename: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """
    Unified transcription function that works with both WhisperX and Azure.
    Automatically selects the transcription service based on:
    1. The 'service' parameter (if provided)
    2. The TRANSCRIPTION_SERVICE environment variable
    3. Availability of Azure credentials (falls back to WhisperX if not available)
    
    Args:
        audio_file (str): Path to audio file
        service (str, optional): Force specific service: "azure" or "whisperx"
        whisper_model (str): WhisperX model to use (only for WhisperX)
        device (str, optional): Device for WhisperX ("cuda", "cpu", "mps")
        enable_diarization (bool): Enable speaker diarization
        save (bool): Whether to save transcription to CSV, default True. set False for live
        **kwargs: Additional service-specific arguments
    Returns:
        dict: {
            "text": str,              # Full transcript
            "segments": List[dict],   # Segments with timestamps
            "service": str,           # azure or whisperx
            "duration": float
        }
    """
    config = TranscriptionConfig()
    
    # Override config
    if service:
        if service.lower() == "azure":
            if not config.azure_key:
                raise ValueError("Azure Speech Service requested but AZURE_SPEECH_KEY not configured")
            use_azure = True
        elif service.lower() == "whisperx":
            use_azure = False
        else:
            raise ValueError(f"Unknown service: {service}. Use 'azure' or 'whisperx'")
    else:
        use_azure = config.use_azure()
    # Normalize audio_files input and process each file
    if not audio_files:
        raise ValueError("audio_files must be provided as a list of file paths")
    if isinstance(audio_files, str):
        audio_files = [audio_files]

    results: List[Dict] = []
    for audio_file in audio_files:
        if os.path.splitext(audio_file)[1].lower() != ".wav":
            wav_file = os.path.splitext(audio_file)[0] + ".wav"
            convert_to_wav(audio_file, wav_file)
            audio_fp = wav_file
        else:
            audio_fp = audio_file

        # Transcribe
        if use_azure:
            print(f"Transcribing (Azure) file: {audio_fp}")
            from transcription.azure_transcribe import transcribe_azure
            result = transcribe_azure(
                audio_file=audio_fp,
                enable_diarization=enable_diarization,
                procedure_type=procedure_type,
                phrase_list=phrase_list,
                save=save,
                save_filename=save_filename if save_filename else "azure_trs.csv",
                **kwargs
            )
            result["service"] = "azure"
        else:
            print(f"Transcribing (WhisperX) file: {audio_fp} using model {whisper_model}")
            from transcription.whisperx_transcribe import transcribe_whisperx
            result = transcribe_whisperx(
                audio_file=audio_fp,
                whisper_model=whisper_model,
                device=device,
                phrase_list=phrase_list,
                procedure_type=procedure_type,
                save_filename=save_filename if save_filename else "whisperx_trs.csv",
                **kwargs
            )
            # transcribe_whisperx should return a dict; if not, normalize it
            if isinstance(result, str):
                result = {"text": result, "segments": [], "duration": 0.0}
            elif isinstance(result, dict):
                if "segments" not in result:
                    result["segments"] = result.get("segments", [])
                if "duration" not in result:
                    result["duration"] = (result.get("segments")[-1].get("end", 0.0)) if result.get("segments") else 0.0
            else:
                result = {"text": str(result), "segments": [], "duration": 0.0}
            result["service"] = "whisperx"

        # attach audio filename and append
        result["audio_file"] = audio_fp
        results.append(result)

    return results

# # Convenience function that maintains backward compatibility
# def transcribe(audio_file: str, **kwargs) -> Dict:
#     """
#     Simple transcribe function with automatic service selection.
#     Drop-in replacement for existing transcribe functions.
#     """
#     res = transcribe_unified(audio_files=[audio_file], **kwargs)
#     return res[0] if res else {"text": "", "segments": [], "duration": 0.0}

if __name__ == "__main__":
    """
    python -m transcription.transcription_service --procedure_type=ercp --audio_file transcription/recordings/ercp/bdstone/bdstone07.m4a
        --service azure
        --service whisperx
    """
    import argparse
    parser = argparse.ArgumentParser(description="Test unified transcription service")
    parser.add_argument("--audio_files", nargs="+", required=True, help="One or more paths to audio files to transcribe")
    parser.add_argument("--service", type=str, choices=["azure", "whisperx"], help="Force specific service (default: auto-detect from config)")
    parser.add_argument("--whisper_model", type=str, default="large-v3", help="WhisperX model (only used with whisperx service)")
    parser.add_argument("--enable_diarization", action="store_true", help="Enable speaker diarization (only supported by Azure)")
    parser.add_argument("--procedure_type", type=str, default=None, help="Procedure type (e.g., ercp, col, egd, eus)")
    parser.add_argument("--save_filename", type=str, default=None, help="Filename to save transcription results (no path). Defaults to azure_trs.csv or whisperx_trs.csv.")
    args = parser.parse_args()
    
    try:
        results = transcribe_unified(
            audio_files=args.audio_files,
            service=args.service,
            whisper_model=args.whisper_model,
            enable_diarization=args.enable_diarization,
            procedure_type=args.procedure_type,
            save_filename=args.save_filename
        )

        # Print final summary
        print(f"\nTRANSCRIPTION COMPLETED for {len(results)} files:")
        print(f"Files transcribed: {', '.join([os.path.basename(r['audio_file']) for r in results])}")
        print(f"Service used: {args.service if args.service else 'auto-detected'}")
        print(f"Saved to: {args.save_filename if args.save_filename else '`service`_trs.csv'}")
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()

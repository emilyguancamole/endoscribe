"""
Unified transcription interface.
Provides a simple way to switch between WhisperX (GPU) and Azure Speech Service.
! not reviewed as of 12/7
"""
import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class TranscriptionConfig:
    """Configuration for transcription service selection."""
    
    # Service types
    WHISPERX = "whisperx"
    AZURE = "azure"
    
    def __init__(self):
        # Default to Azure if credentials are available, otherwise WhisperX
        self.service = os.getenv("TRANSCRIPTION_SERVICE", "azure")
        self.azure_key = os.getenv("AZURE_SPEECH_KEY")
        
        # Auto-detect if Azure is available
        if self.service == "azure" and not self.azure_key:
            print("Warning: AZURE_SPEECH_KEY not found. Falling back to WhisperX.")
            self.service = self.WHISPERX
    
    def use_azure(self) -> bool:
        """Check if Azure should be used."""
        return self.service == self.AZURE and self.azure_key is not None
    
    def use_whisperx(self) -> bool:
        """Check if WhisperX should be used."""
        return self.service == self.WHISPERX


def transcribe_unified(
    audio_file: str,
    service: Optional[str] = None,
    whisper_model: str = "large-v3",
    device: Optional[str] = None,
    enable_diarization: bool = False,
    procedure_type: Optional[str] = None,
    phrase_list: Optional[list] = None,
    **kwargs
) -> Dict:
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
        language (str): Language code (Azure: "en-US", WhisperX: "en")
        enable_diarization (bool): Enable speaker diarization
        **kwargs: Additional service-specific arguments
    
    Returns:
        dict: {
            "text": str,              # Full transcript
            "segments": List[dict],   # Segments with timestamps
            "service": str,           # Service used ("azure" or "whisperx")
            "duration": float         # Audio duration (if available)
        }
    
    Example:
        # Use default service (based on config/environment)
        result = transcribe_unified("audio.wav")
        
        # Force Azure
        result = transcribe_unified("audio.wav", service="azure")
        
        # Force WhisperX with specific model
        result = transcribe_unified("audio.wav", service="whisperx", whisper_model="large-v3")
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
    
    # Perform transcription with selected service
    if use_azure:
        print(f"Transcribing with Azure Speech Service...")
        from transcription.azure_transcribe import transcribe_azure
        result = transcribe_azure(
            audio_file=audio_file,
            enable_diarization=enable_diarization,
            procedure_type=procedure_type,
            phrase_list=phrase_list,
            **kwargs
        )
        result["service"] = "azure"
        
    else:
        print(f"Transcribing with WhisperX (model: {whisper_model})...")
        from transcription.whisperx_transcribe import transcribe_whisperx
        
        result = transcribe_whisperx(
            audio_file=audio_file,
            whisper_model=whisper_model,
            device=device,
            phrase_list=phrase_list,
            procedure_type=procedure_type,
            **kwargs
        )
        result["service"] = "whisperx"
        
        # Add duration if not present
        if "duration" not in result and result.get("segments"):
            result["duration"] = result["segments"][-1].get("end", 0.0)
    
    return result


# Convenience function that maintains backward compatibility
def transcribe(audio_file: str, **kwargs) -> Dict:
    """
    Simple transcribe function with automatic service selection.
    Drop-in replacement for existing transcribe functions.
    """
    return transcribe_unified(audio_file, **kwargs)


if __name__ == "__main__":
    """
    Test unified transcription with both services.
    
    Usage:
        python -m transcription.transcription_service --audio_file path/to/audio.wav
        python -m transcription.transcription_service --audio_file path/to/audio.wav --service azure
        python -m transcription.transcription_service --audio_file path/to/audio.wav --service whisperx
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test unified transcription service")
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--service",
        type=str,
        choices=["azure", "whisperx"],
        help="Force specific service (default: auto-detect from config)"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        help="WhisperX model (only used with whisperx service)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en-US",
        help="Language code (default: en-US for Azure, will be converted for WhisperX)"
    )
    parser.add_argument(
        "--enable_diarization",
        action="store_true",
        help="Enable speaker diarization (only supported by Azure)"
    )
    parser.add_argument(
        "--procedure_type",
        type=str,
        default=None,
        help="Procedure type (e.g., ercp, col, egd, eus)"
    )
    
    args = parser.parse_args()
    
    try:
        result = transcribe_unified(
            audio_file=args.audio_file,
            service=args.service,
            whisper_model=args.whisper_model,
            language=args.language,
            enable_diarization=args.enable_diarization,
            procedure_type=args.procedure_type
        )
        
        print("\n" + "="*80)
        print("TRANSCRIPTION RESULT")
        print("="*80)
        print(f"\nService: {result['service']}")
        print(f"Language: {result['language']}")
        if 'duration' in result:
            print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Segments: {len(result.get('segments', []))}")
        print(f"\nFull Transcript:\n{result['text']}")
        
        if args.enable_diarization and result.get('segments'):
            if any('speaker' in seg for seg in result['segments']):
                print("\n" + "-"*80)
                print("SPEAKER SEGMENTS:")
                print("-"*80)
                for seg in result['segments']:
                    if 'speaker' in seg:
                        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}: {seg['text']}")
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()

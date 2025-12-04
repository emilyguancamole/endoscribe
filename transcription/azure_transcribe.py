"""
Azure Speech Service transcription module.
Provides a drop-in replacement for WhisperX transcription using Azure's managed speech service.
"""
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import subprocess

load_dotenv()

def convert_to_wav(src, dest):
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dest])

def transcribe_azure(
    audio_file: str,
    language: str = "en-US",
    enable_word_level_timestamps: bool = True,
    enable_diarization: bool = False,
    max_speakers: int = 2
) -> Dict[str, any]:
    """
    Transcribe an audio file using Azure Speech Service.
    
    This function provides a similar interface to transcribe_whisperx() for easy migration.
    
    Args:
        audio_file (str): Path to audio file (supports wav, mp3, ogg, flac, etc.)
        language (str): Language code (e.g., "en-US", "en-GB")
        enable_word_level_timestamps (bool): Include word-level timestamps in segments
        enable_diarization (bool): Enable speaker diarization (identifies different speakers)
        max_speakers (int): Maximum number of speakers for diarization (if enabled)
    
    Returns:
        dict: {
            "text": str,  # transcript
            "segments": List[dict],  # List of segments with text, start, end times
            "duration": float  # Audio duration in seconds
        }
    """
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    
    # Configure Azure Speech Service
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )
    speech_config.speech_recognition_language = language
    
    # Enable detailed results with word-level timestamps
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    
    # Configure audio input    
    if os.path.splitext(audio_file)[1].lower() != ".wav":
        wav_file = os.path.splitext(audio_file)[0] + ".wav"
        convert_to_wav(audio_file, wav_file)
        audio_file = wav_file

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    if enable_diarization:
        # Use conversation transcriber for speaker diarization
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )
        conversation_transcriber.properties.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_MaxSpeakerCount,
            str(max_speakers)
        )
        return _transcribe_with_diarization(conversation_transcriber)
    else:
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        return _transcribe_continuous(speech_recognizer)


def _transcribe_continuous(speech_recognizer: speechsdk.SpeechRecognizer) -> Dict:
    """
    Perform continuous recognition without speaker diarization.
    """    
    segments = []
    all_text = []
    done = False
    error_occurred = False
    error_message = None
    
    def recognized_cb(evt):
        """Callback for recognized speech."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            result = evt.result
            segment = {
                "text": result.text,
                "start": result.offset / 10000000.0,  # Convert to seconds
                "end": (result.offset + result.duration) / 10000000.0,
            }
            segments.append(segment)
            all_text.append(result.text)
            print(f"{result.text}")
    
    def canceled_cb(evt):
        """Callback when canceled."""
        nonlocal done, error_occurred, error_message
        print(f"Recognition canceled: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            error_occurred = True
            error_message = f"Error: {evt.error_details}"
            print(error_message)
        done = True
    
    def stopped_cb():
        """Callback when stopped."""
        nonlocal done
        print("Recognition stopped.")
        done = True
    
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.canceled.connect(canceled_cb)
    speech_recognizer.session_stopped.connect(stopped_cb)
    
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.5)
    speech_recognizer.stop_continuous_recognition()
    if error_occurred:
        raise RuntimeError(error_message)
    
    # Calculate duration
    duration = segments[-1]["end"] if segments else 0.0
    
    # Join text with single quote separator #! currently matching WhisperX format, todo change
    full_text = " '".join(all_text).replace("  ", " ").strip()

    print(f"Transcription complete. Duration: {duration:.2f}s, Segments: {len(segments)}")
    return {
        "text": full_text,
        "segments": segments,
        "duration": duration
    }

def _transcribe_with_diarization(
    conversation_transcriber: speechsdk.transcription.ConversationTranscriber
) -> Dict:
    """
    Perform transcription with speaker diarization.
    """
    print("Starting Azure Speech transcription with speaker diarization...")
    
    segments = []
    all_text = []
    done = False
    error_occurred = False
    error_message = None
    
    def transcribed_cb(evt):
        """Callback for transcribed speech with speaker info."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            result = evt.result
            speaker_id = result.speaker_id if hasattr(result, 'speaker_id') else "Unknown"
            segment = {
                "text": result.text,
                "start": result.offset / 10000000.0,
                "end": (result.offset + result.duration) / 10000000.0,
                "speaker": speaker_id
            }
            
            segments.append(segment)
            all_text.append(result.text)
            print(f"[Speaker {speaker_id}] {result.text}")
    
    def canceled_cb(evt):
        """Callback for canceled recognition."""
        nonlocal done, error_occurred, error_message
        print(f"Recognition canceled: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            error_occurred = True
            error_message = f"Error: {evt.error_details}"
            print(error_message)
        done = True
    
    def stopped_cb(evt):
        """Callback when recognition stops."""
        nonlocal done
        print("Recognition stopped.")
        done = True
    
    conversation_transcriber.transcribed.connect(transcribed_cb)
    conversation_transcriber.canceled.connect(canceled_cb)
    conversation_transcriber.session_stopped.connect(stopped_cb)
    
    conversation_transcriber.start_transcribing_async().get()
    while not done:
        time.sleep(0.5)
    
    conversation_transcriber.stop_transcribing_async().get()
    
    if error_occurred:
        raise RuntimeError(error_message)
    
    duration = segments[-1]["end"] if segments else 0.0
    
    #! Join text with single quote separator (matching WhisperX format)
    full_text = " '".join(all_text).replace("  ", " ").strip()
    
    print(f"Transcription complete. Duration: {duration:.2f}s, Segments: {len(segments)}")
    
    return {
        "text": full_text,
        "segments": segments,
        "duration": duration
    }

def transcribe(audio_file: str, **kwargs) -> Dict:
    """
    Simple wrapper function with same signature as whisperx transcribe functions.
    
    Args:
        audio_file (str): Path to audio file
        **kwargs: Additional arguments passed to transcribe_azure
    
    Returns:
        dict: Transcription result with "text" and "segments" keys
    """
    return transcribe_azure(audio_file, **kwargs)


if __name__ == "__main__":
    """
    Test the Azure transcription service.
    
    Usage:
        python -m transcription.azure_transcribe --audio_file path/to/audio.wav
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Azure Speech Service transcription")
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--enable_diarization",
        action="store_true",
        help="Enable speaker diarization"
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=2,
        help="Maximum number of speakers for diarization (default: 2)"
    )
    
    args = parser.parse_args()
    
    try:
        result = transcribe_azure(
            audio_file=args.audio_file,
            language="en-US",
            enable_diarization=args.enable_diarization,
            max_speakers=args.max_speakers
        )
        
        print("\n" + "="*80)
        print("TRANSCRIPTION RESULT")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Segments: {len(result['segments'])}")
        print(f"\nFull Transcript:\n{result['text']}")
        
        if args.enable_diarization:
            print("\n" + "-"*80)
            print("SPEAKER SEGMENTS:")
            print("-"*80)
            for seg in result['segments']:
                speaker = seg.get('speaker', 'Unknown')
                print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {speaker}: {seg['text']}")
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()

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
    max_speakers: int = 2,
    phrase_list: Optional[List[str]] = None,
    procedure_type: Optional[str] = None,
    save_filename: Optional[str] = 'azure_trs.csv',
) -> Dict[str, any]:
    """
    Transcribe an audio file using Azure Speech Service. Similar interface to transcribe_whisperx() for migration.
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
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )
    speech_config.speech_recognition_language = language
    
    # Enable detailed results with word-level timestamps
    # speech_config.request_word_level_timestamps()
    # speech_config.output_format = speechsdk.OutputFormat.Detailed
    
    # Configure audio input    
    if os.path.splitext(audio_file)[1].lower() != ".wav":
        wav_file = os.path.splitext(audio_file)[0] + ".wav"
        convert_to_wav(audio_file, wav_file)
        audio_file = wav_file

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    print(f"Starting Azure Speech transcription.")
    if save_filename or procedure_type:
        print(f"  Will save to transcription/results/{procedure_type}/{save_filename}.csv")

    # If no phrase_list explicitly provided, try to load one based on procedure_type
    if phrase_list is None and procedure_type:
        phrase_list = _load_phrase_list_for_procedure(procedure_type)

    if enable_diarization: # conversation transcriber for speaker diarization
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
        conversation_transcriber.properties.set_property(speechsdk.PropertyId.SpeechServiceConnection_MaxSpeakerCount, str(max_speakers))
        result = _transcribe_with_diarization(conversation_transcriber, phrase_list=phrase_list)
    else:
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = _transcribe_continuous(speech_recognizer, phrase_list=phrase_list)

    if procedure_type or save_filename:
        try:
            _save_transcription_result(result, audio_file, save_filename, procedure_type)
        except Exception as e:
            print(f"Warning: failed to save transcription result: {e}")

    return result

def _transcribe_continuous(speech_recognizer: speechsdk.SpeechRecognizer, phrase_list: Optional[List[str]] = None) -> Dict:
    """
    Perform continuous transcription without speaker diarization.
    """    
    segments = []
    all_text = []
    done = False
    error_occurred = False
    error_message = None
    
    # Apply phrase list grammar if provided (improves recognition for domain-specific terms)
    if phrase_list:
        try:
            grammar = speechsdk.PhraseListGrammar.from_recognizer(speech_recognizer)
            for p in phrase_list:
                if p:
                    grammar.addPhrase(str(p))
            print(f"Applied {len(phrase_list)} phrases to recognizer phrase list grammar")
        except Exception as e:
            print(f"Warning: could not apply phrase list to recognizer: {e}")

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
    
    def stopped_cb(evt):
        """Callback when stopped. Note needs the param for Azure """
        nonlocal done
        try:
            print("Recognition stopped.")
        except Exception as e:
            print(f"Error in stopped_cb print: {e}")
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
    
    duration = segments[-1]["end"] if segments else 0.0
    full_text = " ".join(all_text).replace("  ", " ").strip()

    return {
        "text": full_text,
        "segments": segments,
        "duration": duration
    }

def _transcribe_with_diarization(
    conversation_transcriber: speechsdk.transcription.ConversationTranscriber,
    phrase_list: Optional[List[str]] = None,
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
    
    # Try to apply phrase list grammar to conversation transcriber if supported
    if phrase_list:
        try:
            # Some SDKs provide a from_transcriber helper; attempt and fall back gracefully
            if hasattr(speechsdk.PhraseListGrammar, 'from_transcriber'):
                grammar = speechsdk.PhraseListGrammar.from_transcriber(conversation_transcriber)
            else:
                # Fallback: try from_recognizer (may not affect conversation transcriber)
                grammar = None
            if grammar is not None:
                for p in phrase_list:
                    if p:
                        grammar.addPhrase(str(p))
                print(f"Applied {len(phrase_list)} phrases to conversation transcriber grammar")
            else:
                print("Phrase list not applied to conversation transcriber (no supported API available)")
        except Exception as e:
            print(f"Warning: could not apply phrase list to conversation transcriber: {e}")

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


def _load_phrase_list_for_procedure(procedure_type: str) -> List[str]:
    """Load a simple phrase list for a given procedure type. Looks for `prompts/{procedure_type}/phrases.txt` or `prompts/{procedure_type}_phrases.txt`.
//? idk revisit
    """
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    candidates = [
        os.path.join(base, procedure_type, "phrases.txt"),
        os.path.join(base, f"{procedure_type}_phrases.txt"),
        os.path.join(base, procedure_type, "phrases.csv"),
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    phrases = [line.strip() for line in fh if line.strip()]
                    if phrases:
                        print(f"Loaded {len(phrases)} phrases from {p}")
                        return phrases
        except Exception:
            continue
    fallback = {
        "ercp": ["sphincterotomy", "choledocholithiasis", "common bile duct", "pancreatic duct",
            "cannulation", "sphincter of Oddi", "endoscopic retrograde cholangiopancreatography",
            "ERCP", "stent", "biliary stent", "pancreatitis"],
        "col": ["polyp", "polypectomy", "cecum", "sigmoid", "ileocecal", "adenoma"],
        "egd": ["gastroscopy", "esophagus", "stomach", "duodenum", "ulcer", "biopsy"],
        "eus": ["endoscopic ultrasound", "pancreas", "biliary", "lesion"],
    }
    return fallback.get(procedure_type.lower(), [])


def _save_transcription_result(result: Dict, audio_file: str, save_filename: Optional[str], procedure_type: Optional[str]) -> None:
    """
    Save transcription into `transcription/results/{procedure_type}/save_filename` if provided, or `transcription/results/{procedure_type}/transcriptions.csv`.
    If `save_filename` is provided but no procedure_type, saves under `transcription/results/_misc/`.
    """
    repo_root = os.path.dirname(os.path.dirname(__file__))
    proc_folder = procedure_type if procedure_type else "_misc"
    results_dir = os.path.join(repo_root, "transcription", "results", proc_folder)
    os.makedirs(results_dir, exist_ok=True)

    out_fp = os.path.join(results_dir, save_filename if save_filename.lower().endswith(".csv") else f"{save_filename}.csv")
    write_header = not os.path.exists(out_fp)
    import csv
    file_id = os.path.splitext(os.path.basename(audio_file))[0]
    row = {
        "file": file_id,
        "procedure_type": procedure_type,
        "audio_fp": audio_file,
        "pred_transcript": result.get("text", "")
    }
    with open(out_fp, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "procedure_type", "audio_fp", "pred_transcript"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved transcription to {out_fp}")

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
        python -m transcription.azure_transcribe --audio_file path/to/audio.wav
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Azure Speech Service transcription")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to audio file to transcribe")
    parser.add_argument("--enable_diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--max_speakers", type=int,default=2, help="Maximum number of speakers for diarization (default: 2)")
    parser.add_argument("--procedure_type", type=str, default=None, help="Procedure type (ercp, col, egd, eus)")
    parser.add_argument("--save_filename", type=str, default=None, help="Filename to save transcription result (optional)")
    args = parser.parse_args()

    phrases = [
        "endoscopy",
        "sphincterotomy",
        "choledocholithiasis",
        "cannulation",
        "common bile duct",
        "pancreatic duct",
        "cannulation",
        "sphincter of Oddi",
        "endoscopic retrograde cholangiopancreatography",
        "ERCP",
        "stent",
        "biliary stent",
        "pancreatitis"
    ]
    
    try:
        result = transcribe_azure(
            audio_file=args.audio_file,
            language="en-US",
            enable_diarization=args.enable_diarization,
            max_speakers=args.max_speakers,
            phrase_list=phrases,
            procedure_type=args.procedure_type,
            save_filename=args.save_filename if hasattr(args, 'save_filename') else None,
        )
        print("\n" + "="*80)
        print("TRANSCRIPTION RESULT")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Segments: {len(result['segments'])}")
        print(f"\nPreview:\n{result['text'][:200]}...")
        
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

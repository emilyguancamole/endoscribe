###################### https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate
import argparse
import os
import pandas as pd
import torch

# Dynamic CUDA configuration - only set if CUDA is available
if torch.cuda.is_available():
    # Use CUDA_VISIBLE_DEVICES env var if set, otherwise use default GPUs for WhisperX
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,8,9"
        print(f"CUDA detected. Setting CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("CUDA not available. WhisperX will run on CPU or MPS (Apple Silicon).")

import noisereduce as nr
import soundfile as sf
import pandas as pd
try:
    from transcription.convert_to_mono import batch_convert
except ModuleNotFoundError:
    from convert_to_mono import batch_convert
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
import whisperx 

def suppress_noise(input_audio_fp: str, output_audio_path: str):
    audio, sr = sf.read(input_audio_fp)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    sf.write(output_audio_path, reduced_noise, sr)
    return output_audio_path

def transcribe(audio_file, whisper_model="large-v3", device=None):
    """
    Transcribe an audio file using WhisperX with speaker diarization.
        NOTES: do not need to manually tokenize or preprocess audio; WhisperX handles it.

    Args:
        audio_file (str): Path to audio file
        whisper_model (str): WhisperX model name (e.g., "large-v3")
        device (str): Device to use ("cuda", "cpu", "mps"). If None, auto-detects.

    Returns:
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {device}")

    model = whisperx.load_model(whisper_model, device=device, compute_type="float16")
    audio = whisperx.load_audio(audio_file)

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=8, language="en")

    # Align whisper output to improve word level timing alignment
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    #todo add Speaker diarization; uses pyannote/pyannote-audio
    # print("Running diarization...")
    # diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
    # diarize_segments = diarize_model(audio)
    # result = whisperx.assign_word_speakers(diarize_segments, result) # combine transcipt with diarization
    # print("Result segments:\n", print(result['segments']))

    # Main speaker
    # first_seconds = [seg for seg in result['segments'] if seg['start'] < 60.0] # first n secs
    # speaker_counts = Counter([seg['speaker'] for seg in first_seconds]) # count speakers in first seconds
    # main_speaker = speaker_counts.most_common(1)[0][0] if speaker_counts else 0
    # print(f"Identified main speaker as: {main_speaker}")
    # for seg in result['segments']:
    #     if 'speaker' not in seg:
    #         print("Missing speaker in seg:", seg)
    # speaker_segments = [seg for seg in result['segments'] if seg.get('speaker') == main_speaker] 
    # speaker_transcript = " ".join([seg['text'] for seg in speaker_segments])
    # bkg_segments = [seg for seg in result['segments'] if seg.get('speaker') != main_speaker]
    # bkg_transcript = " ".join([seg['text'] for seg in bkg_segments])
    # print("SPEAKER TRANSCRIPT:\n", speaker_transcript)
    # print("BKG :\n", bkg_segments)
    result_text = [seg['text'] for seg in result['segments']] 
    # join into one string but leave single quotes between segments
    result_text = " '".join(result_text).replace("  ", " ").strip()
    print("Final transcript:\n", result_text)
    return result_text


def transcribe_whisperx(audio_file, whisper_model="large-v3", device=None):
    """
    Transcribe an audio file using WhisperX with alignment. based on above old transcribe() function
    Returns the full text and segment details.

    USED IN PEP_RISK -- #todo make normal endoscribe transcription use this function

    Args:
        audio_file (str): Path to audio file
        whisper_model (str): WhisperX model name (e.g., "large-v3")
        device (str): Device to use ("cuda", "cpu", "mps"). If None, auto-detects.
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {device}")

    print(f"Loading WhisperX model: {whisper_model}")
    model = whisperx.load_model(whisper_model, device=device, compute_type="float16")

    print(f"Loading audio: {audio_file}")
    audio = whisperx.load_audio(audio_file)

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=8, language="en")
    # print("Transcription result:\n", result)

    # Alignment
    print("Aligning transcription for better accuracy...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # Join all segments into one text string, separated by single quote
    result_text = " '".join(seg["text"] for seg in aligned_result["segments"]).replace("  ", " ").strip()

    return {
        "text": result_text,
        "segments": aligned_result["segments"]
    }


def get_procedure_type(proc_row):
    # procedure_type is the 'true' column in procedures_df: procedure_type_egd,procedure_type_eus,procedure_type_ercp,procedure_type_colonoscopy,procedure_type_endoflip,procedure_type_sigmoidoscopy -> #? can there be more than 1 procedure type
        # else use the text value written in procedure_type_other
    if proc_row.empty:
        return "unknown"
    for proc in ['egd', 'eus', 'ercp', 'colonoscopy', 'endoflip', 'sigmoidoscopy']:
        if proc_row[f'procedure_type_{proc}'].values[0]:
            return proc if proc != 'colonoscopy' else 'col'
    return proc_row['procedure_type_other'].values[0].lower() if pd.notna(proc_row['procedure_type_other'].values[0]) else 'unknown'


if __name__ == "__main__":
    '''
    Run from endoscribe!! in IA1
    
    python -m transcription.whisperx_transcribe \
    --convert_to_mono \
    --save_filename=long-10-2025 \
    --model=large-v3 \
    --audio_dir=transcription/recordings/long \
    --procedures_data=data/procedures.csv

    Notes: 
        - for whisperx, model names don't have "openai/" or "whisper-" prefix
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to audio folder (general, not specialized vocals folder)')
    parser.add_argument('--convert_to_mono', action='store_true', help='Flag to convert all audio files in the audio_dir to mono')
    parser.add_argument('--save_filename', type=str, required=True, help='Name of file, without .csv, to save transcriptions to')
    parser.add_argument('--procedures_data', default="data/procedures.csv", help="Path to procedures data csv")
    parser.add_argument('--model', type=str, required=True) #'distil-whisper/distil-large-v3' #openai/whisper-medium.en 
    parser.add_argument('--use_prompt', default=False, help='Whether to use the provided prompt for each audio file')
    args = parser.parse_args()

    if args.convert_to_mono:
        print(f"Converting all audio files in {args.audio_dir} to mono...")
        batch_convert(args.audio_dir, inp_audio_format="wav")

    output_fp = f"transcription/results/longform/{args.save_filename}.csv" # longform directory for mixed procedure types

    if os.path.exists(output_fp):
        transcribed_df = pd.read_csv(output_fp)
        print("Loaded existing transcription df")
    else:
        transcribed_df = pd.DataFrame(columns=["participant_id", "procedure_type", "audio_dir", "notes", "pred_transcript"])
        # transcribed_df = pd.DataFrame(columns=["file", "notes", "pred_transcript"])
    transcribed_df['participant_id'] = transcribed_df['participant_id'].astype(str)
    transcribed_filenames = set(transcribed_df['participant_id']) 
    print("with already transcribed files:", transcribed_filenames)

    procedures_df = pd.read_csv(args.procedures_data) # used for getting procedure type
    procedures_df['participant_id'] = procedures_df['participant_id'].astype(str)
  
    vocals_dir = args.audio_dir
    print("Using model: ", args.model)

    # Transcribe each audio vocal file in directory
    for filename in os.listdir(vocals_dir):
        if filename == ".DS_Store": continue

        participant_id = str('.'.join(filename.split("_")[1].split(".")[:-1]))
        print("case name:", participant_id)
        if participant_id in transcribed_filenames:
            print(f"Skipping {participant_id}, already transcribed")
            continue

        # Infer procedure type by finding participant_id in procedures_df
        proc_row = procedures_df[procedures_df['participant_id'] == participant_id]
        procedure_type = get_procedure_type(proc_row)
        print(f"Inferred procedure type as: {procedure_type}")

        audio_fp = os.path.join(vocals_dir, filename)
        print(f"Transcribing {audio_fp}; use prompt: {args.use_prompt}")
        transcript = transcribe_whisperx(audio_fp, args.model)['text']
        transcribed_df.loc[len(transcribed_df)] = [participant_id, procedure_type, audio_fp, "", transcript]

        transcribed_df.to_csv(output_fp, index=False) # Save intermediate results

    print(f"Saved all procedure transcriptions to {args.save_filename}")

    # Split into different procedure types and also save as separate files in respective folders. 
    # note will overwrite existing files
    for proc in transcribed_df['procedure_type'].unique():
        proc_df = transcribed_df[transcribed_df['procedure_type'] == proc]
        proc_output_fp = f"transcription/results/{proc}/{args.save_filename}.csv"
        proc_df.to_csv(proc_output_fp, index=False)
        print(f"Saved {len(proc_df)} {proc} transcriptions to {proc_output_fp}")

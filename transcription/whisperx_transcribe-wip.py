###################### https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,9"
import noisereduce as nr
import soundfile as sf
import pandas as pd
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

def transcribe(audio_file, whisper_model="large-v3", device="cuda"):
    """
    Transcribe an audio file using WhisperX with speaker diarization.
        NOTES: do not need to manually tokenize or preprocess audio; WhisperX handles it.
    
    Args:
        audio_file (str): Path to audio file
        whisper_model (str): WhisperX model name (e.g., "large-v3")
        device (str): "cuda" or "cpu"
    
    Returns:
    """

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


if __name__ == "__main__":
    '''
    Run from endoscribe!! in IA1
    
    python transcription/whisperx_transcribe-wip.py \
    --save_dir=transcription/results/ercp --save_filename=long_9-30-2025 \
    --model=large-v3 \
    --audio_dir=transcription/recordings/long

        --convert_to_mono \
    Notes: 
        - model names don't have "openai/" or "whisper-" prefix
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--audio_dir', type=str, required=True, help='Path to audio folder (general, not specialized vocals folder)')
    argparser.add_argument('--convert_to_mono', action='store_true', help='Flag to convert all audio files in the audio_dir to mono')
    argparser.add_argument('--save_dir', type=str, required=True)
    argparser.add_argument('--save_filename', type=str, required=True, help='Name of file, without .csv, to save transcriptions to')
    argparser.add_argument('--datasheet_fp', type=str,  default="")
    argparser.add_argument('--model', type=str, required=True) #'distil-whisper/distil-large-v3' #openai/whisper-medium.en 
    argparser.add_argument('--use_prompt', default=False, help='Whether to use the provided prompt for each audio file')
    args = argparser.parse_args()

    if args.convert_to_mono:
        print(f"Converting all audio files in {args.audio_dir} to mono...")
        batch_convert(args.audio_dir)

    output_fp = f"{args.save_dir}/{args.save_filename}.csv"

    if os.path.exists(output_fp):
        transcribed_df = pd.read_csv(output_fp)
        print("Loaded existing transcription df")
    else:
        transcribed_df = pd.DataFrame(columns=["file", "notes", "pred_transcript"])
    transcribed_df['file'] = transcribed_df['file'].astype(str)
    transcribed_filenames = set(transcribed_df['file']) 
    print("with already transcribed files:", transcribed_filenames)

  
    vocals_dir = args.audio_dir

    print("Using model: ", args.model)

    # Transcribe each audio vocal file in directory.
    for filename in os.listdir(vocals_dir):
        if filename == ".DS_Store":
            continue

        case_name = str(filename.split(".")[0])

        #! CURRENTLY OVERWRITING If file already transcribed, skip
        # if case_name in transcribed_filenames:
        #     print(f"Skipping {case_name}, already transcribed")
        #     continue

        #todo Noise suppression (optional) - creates a new file with _cleaned suffix
        # audio_fp = suppress_noise(os.path.join(vocals_dir, filename), os.path.join(vocals_dir, filename.split(".")[0] + "_cleaned.wav"))

        audio_fp = os.path.join(vocals_dir, filename)
        print(f"Transcribing {audio_fp}; saving to {args.save_dir}/{args.save_filename}")
        print(f"    Use prompt: {args.use_prompt}")
        
        transcript = transcribe(audio_fp, args.model)
        transcribed_df.loc[len(transcribed_df)] = [case_name, "", transcript]

        transcribed_df.to_csv(output_fp, index=False) # Save intermediate results

    print(f"Saved transcriptions to {args.save_dir}")

###################### https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import subprocess as sp
import io
from typing import Dict, Tuple, Optional, IO
from pathlib import Path
import sys
import shutil
import select
import re
import pandas as pd
import torch
import librosa
from datasets import load_dataset
import evaluate
from convert_to_mono import batch_convert
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

import whisperx                     # https://github.com/m-bain/whisperX
from collections import Counter

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

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


    # Speaker diarization; uses pyannote/pyannote-audio
    print("Running diarization...")
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result) # align results with diarization
    # print("result segments:\n", print(result['segments']))
    first_seconds = [seg for seg in result['segments'] if seg['start'] < 60.0] # first n secs
    speaker_counts = Counter([seg['speaker'] for seg in first_seconds]) # count speakers in first seconds
    main_speaker = speaker_counts.most_common(1)[0][0] if speaker_counts else 0
    print(f"Identified main speaker as: {main_speaker}")

    for seg in result['segments']:
        if 'speaker' not in seg:
            print("Missing speaker in seg:", seg)
    speaker_segments = [seg for seg in result['segments'] if seg.get('speaker') == main_speaker] 
    speaker_transcript = " ".join([seg['text'] for seg in speaker_segments])

    bkg_segments = [seg for seg in result['segments'] if seg.get('speaker') != main_speaker]
    bkg_transcript = " ".join([seg['text'] for seg in bkg_segments])
    
    print("SPEAKER TRANSCRIPT:\n", speaker_transcript)
    print("BKG :\n", bkg_segments)
    return speaker_transcript, bkg_segments



if __name__ == "__main__":
    '''
    Run from endoscribe!!
    
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/colonoscopy_results --save_filename=lg_45 \
    --model=/scratch/eguan2/whisper_lg_45 \
    --audio_dir=transcription/recordings/finetune_testset

    Colonoscopy with Errors:
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/colonoscopy_results/with_errorsconvo --save_filename=whisper_lg_v3 \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/colonoscopy/with_errorsconvo

    EUS (cyst):
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/eus_results --save_filename=whisper_lg_v3_clean_noprompt \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/eus/cyst

    ERCP
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/ercp_results --save_filename=whisper_lg_v3 \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/ercp/bdstone

    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--audio_dir', type=str, required=True, help='Path to audio folder (general, not specialized vocals folder)')
    argparser.add_argument('--save_dir', type=str, required=True)
    argparser.add_argument('--save_filename', type=str, required=True, help='Name of file, without .csv, to save transcriptions to')
    argparser.add_argument('--datasheet_fp', type=str,  default="")
    argparser.add_argument('--do_separate_vocals', type=str,  default=False, help='Whether to run DEMUCS vocal separation on audio recordings')
    argparser.add_argument('--model', type=str, required=True) #'distil-whisper/distil-large-v3' #openai/whisper-medium.en #openai/whisper-large-v3') 
    argparser.add_argument('--use_prompt', default=False, help='Whether to use the provided prompt for each audio file')
    argparser.add_argument('--test_labels_fp', 
                            default='transcription/finetune_data/test_labels.csv') # CSV of gold transcripts, columns=[file, transcript]
    args = argparser.parse_args()

    output_fp = f"{args.save_dir}/{args.save_filename}.csv"

    do_transcribe = True # I can toggle these
    do_evaluate_results = False

    ############ TRANSCRIBE ##############
    if do_transcribe:
        if os.path.exists(output_fp):
            transcribed_df = pd.read_csv(output_fp)
            print("Loaded existing transcription df")
        else:
            transcribed_df = pd.DataFrame(columns=["file", "notes", "pred_transcript"])
        transcribed_df['file'] = transcribed_df['file'].astype(str)
        transcribed_filenames = set(transcribed_df['file']) 
        print("with already transcribed files:", transcribed_filenames)

        bkg_transcript_df = pd.read_csv(f"{args.save_dir}/{args.save_filename}_bkg.csv")

        #! Separate vocals or not - separate is outdated
        if args.do_separate_vocals: # if not os.path.exists(f"{args.audio_dir}/htdemucs"):
            # separate_vocals(args.audio_dir, output_audio_path=args.audio_dir) # creates a subfolder `htdemucs` in the output path
            # vocals_dir = f"{args.audio_dir}/vocals"
            # print("vocals_dir", vocals_dir) # folder with the actual files to transcribe
            # move_vocals_files(vocals_dir)
            print("vocal separation not implemented")
        else:
            vocals_dir = args.audio_dir

        print("Using model: ", args.model)

        # Transcribe each audio vocal file in directory.
        for filename in os.listdir(vocals_dir):
            if filename == ".DS_Store":
                continue

            case_name = str(filename.split(".")[0])

            #! If file already transcribed, skip
            # if case_name in transcribed_filenames:
            #     print(f"Skipping {case_name}, already transcribed")
            #     continue
            
            # Transcribe, separate speaker and background
            audio_fp = os.path.join(vocals_dir, filename)
            print(f"Transcribing {audio_fp} with use_prompt={args.use_prompt}; saving to {args.save_dir}/{args.save_filename}")
            
            speaker_transcript, bkg_transcript = transcribe(audio_fp, args.model)
            transcribed_df.loc[len(transcribed_df)] = [case_name, "", speaker_transcript[0]]
            bkg_transcript_df.loc[len(bkg_transcript_df)] = [case_name, "", bkg_transcript[0]]

            transcribed_df.to_csv(output_fp, index=False) # Save intermediate results
            bkg_transcript_df.to_csv(f"{args.save_dir}/{args.save_filename}_bkg.csv", index=False)

        print(f"Saved transcriptions to {args.save_dir}")

        

    ############# Evaluation - outdated! ###################
    # if do_evaluate_results:
    #     print("Evaluation of resulting transcript:")
    #     processed_pred_df = process_predictions(output_fp)
    #     wer_whisper = compute_metrics(processed_pred_df, 
    #                                 labels=args.test_labels_fp)
    #     print(f"WER for {args.model}: {wer_whisper}")


    print("Done")
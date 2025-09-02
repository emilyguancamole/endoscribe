###################### https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
import subprocess as sp
import io
from typing import Dict, Tuple, Optional, IO
from pathlib import Path
import sys
import shutil
import select
import pandas as pd
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from .convert_to_mono import batch_convert
from .utils import process_predictions, compute_metrics


def copy_process_streams(process: sp.Popen):
    '''from DEMUCS Google colab. Repo: https://github.com/adefossez/demucs'''
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

def separate_vocals(input_audio_path, output_audio_path):
    '''With model "htdemucs". Ensure DEMUCS is installed `python3 -m pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs`
    Code based on DEMUCS Google colab. Repo: https://github.com/adefossez/demucs
    '''
    # if input file is not wav, convert to wav and overwrite
    # if not input_audio_path.endswith(".wav") or not input_audio_path.endswith(".WAV"):
        
    original_format = "m4a" # so far, only this
    batch_convert(input_audio_path, original_format, replace=True)

    print(input_audio_path, output_audio_path)
    cmd = ["python3", "-m", "demucs.separate", "-o", str(output_audio_path), "-n", "htdemucs"]
    
    files = [f for f in Path(args.audio_dir).iterdir() if f.is_file()]

    print("Files to separate:", files)
    print("With command: ", " ".join(cmd))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")

def move_vocals_files(vocals_dir):
    # Move vocals files to a new location vocals/{filename}_vocals -- transcription should use this
    htdemucs_dir = os.path.join(args.audio_dir, "htdemucs")
    if not os.path.exists(vocals_dir):
        os.makedirs(vocals_dir)

    for filename in os.listdir(htdemucs_dir):
        subfolder_path = os.path.join(htdemucs_dir, filename)
        if os.path.isdir(subfolder_path):
            vocals_file = os.path.join(subfolder_path, "vocals.wav")
            if os.path.exists(vocals_file):
                dest_path = os.path.join(vocals_dir, f"{filename}_vocals.wav")
                shutil.move(vocals_file, dest_path)
            else:
                print(f"Skipping {subfolder_path}, no vocals.wav found")
    print(f"All available vocals.wav files moved to {vocals_dir}")

def transcribe(audio_file, whisper_model, prompt=None) -> str:
    """Given a (optional) prompt, transcribe the given audio file."""
     
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
    # model = whisperx.load_model(whisper_model, device="cuda")

    # Using pre-trained tokenizer# tokenizer = WhisperTokenizer.from_pretrained(whisper_model, language='English', task="transcribe")
    processor = WhisperProcessor.from_pretrained(whisper_model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)

    # Use the model and processor to transcribe the audio. 
    audio, sr = librosa.load(audio_file, sr=16000) # librosa returns the audio as a numpy array and its sampling rate.
    inputs = processor(
        audio, sampling_rate=sr, 
        return_tensors="pt", # Return PyTorch torch.Tensor objects.
        truncation=False, 
        padding="longest", 
        return_attention_mask=True,
    ).input_features
    if inputs.shape[-1] < 3000:
        # we in-fact have short-form -> pre-process accordingly
        inputs = processor(
            audio, sampling_rate=sr,
            return_tensors="pt",
        ).input_features

    decode_options = dict(language="en", 
                        num_beams=5, 
                        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                        compression_ratio_threshold=2.24,#1.35
                        # logprob_threshold=-1.0,
                        # no_speech_threshold=0.1,
                        # condition_on_prev_tokens=True,
                        )
    transcribe_options = dict(task="transcribe", **decode_options)

    if prompt:
        print("Using initial prompt")
        prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt")
        predicted_ids = model.generate(inputs,
            prompt_ids=prompt_ids,
            **transcribe_options,
            ) 
        transcription = processor.batch_decode(predicted_ids, prompt_ids=prompt_ids, skip_special_tokens=True)[0]   
    else:
        print("No prompt")
        predicted_ids = model.generate(inputs,
            **transcribe_options,
            return_timestamps=True
            ) 
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]   

    print("TRANSCRIPT:\n",transcription)
    return transcription


def move_to_datasheet(transcriptions_df, datasheet_fp):
    # move transcriptions to /Users/emilyguan/Downloads/EndoScribe/datasheets/Sample_1_test_datasheet_finetune.csv. match by column 'file'
    datasheet_df = pd.read_csv(datasheet_fp) #, encoding='ISO-8859-1'
    print(transcribed_df.columns)
    print(datasheet_df.columns)
    datasheet_df = datasheet_df.merge(transcriptions_df, on='file', how='left')
    datasheet_df.to_csv(datasheet_fp, index=False)
    print("Saved to datasheet", datasheet_fp)


if __name__ == "__main__":
    '''
    Run from endoscribe!!
    
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/results/col --save_filename=lg_45 \
    --model=/scratch/eguan2/whisper_lg_45 \
    --audio_dir=transcription/recordings/finetune_testset

    Colonoscopy with Errors:
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/results/col/with_errorsconvo --save_filename=whisper_lg_v3 \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/colonoscopy/with_errorsconvo

    EUS (cyst):
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/results/eus --save_filename=whisper_lg_v3_clean_noprompt \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/eus/cyst

    ERCP
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/results/ercp --save_filename=whisper_lg_v3 \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/ercp/bdstone

    EGD
    python transcription/whisper_transcribe.py \
    --save_dir=transcription/results/egd --save_filename=whisper_lg_v3 \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/egd

    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--audio_dir', type=str, required=True, help='Path to audio folder (general, not specialized vocals folder)')
    argparser.add_argument('--save_dir', type=str, required=True)
    argparser.add_argument('--save_filename', type=str, required=True, help='Name of file, without .csv, to save transcriptions to')
    argparser.add_argument('--datasheet_fp', type=str,  default="")
    argparser.add_argument('--do_separate_vocals', type=str,  default=False, help='Whether to run DEMUCS vocal separation on audio recordings')
    argparser.add_argument('--model', type=str, required=True) #'distil-whisper/distil-large-v3' #openai/whisper-medium.en #openai/whisper-large-v3') 
    argparser.add_argument('--use_prompt', default=False)
    argparser.add_argument('--test_labels_fp', 
                            default='transcription/finetune_data/test_labels.csv') # CSV of gold transcripts, columns=[file, transcript]
    args = argparser.parse_args()

    output_fp = f"{args.save_dir}/{args.save_filename}.csv"

    # Colonoscopy prompt
    # prompt = "Boston Bowel Prep, colonoscopy, polyp, transverse colon, cold snare, Roth net, 2 mm, sessile, NICE classification, Paris classification IIa, JNET, cecum, LST-G, endoclips, swift coagulation, submucosal fibrosis, tattoo, retroflexion, diverticulosis"
    # EUS prompt
    # prompt = "SharkCore, Dr. Venkat Akshintala, Bayview Medical Center, endoscopic, portal confluence, duodenum, mucosa, pancreas, parenchyma, FNA, FNB, LA Grade esophagitis, focal lobularity, nodular deposits, major papilla, strictures, echogenic foci, duodenal secretion"
    # ERCP prompt
    # prompt = "Dr. Venkat Akshintala, Bayview Medical Center, ERCP, timeout, duodenum, duodenoscope, scout film, biliary cannulation, cannulate, SpyBite biopsy, rat tooth forceps, esophagus, bile duct stricture, sphincterotome, stone retrieval balloon, stent, French, Visiglide wire, cholangiogram, cholangioscopy, papilla, pancreatic genu, Soehendra stent, Cook Zimmon double pigtail"

    do_transcribe, do_evaluate_results = True, False # I can toggle these

    ############ TRANSCRIBE ##############
    if do_transcribe:
        # Check if the file already exists 
        if os.path.exists(output_fp):
            transcribed_df = pd.read_csv(output_fp)
            print("Loaded existing transcription df")
        else:
            transcribed_df = pd.DataFrame(columns=["file", "notes", "pred_transcript"])
        
        transcribed_df['file'] = transcribed_df['file'].astype(str)
        transcribed_filenames = set(transcribed_df['file']) 
        print("with already transcribed files:", transcribed_filenames)

        #!outdated, return later. Separate vocals or not
        if args.do_separate_vocals: 
            # separate_vocals(args.audio_dir, output_audio_path=args.audio_dir) # creates a subfolder `htdemucs` in the output path
            # vocals_dir = f"{args.audio_dir}/vocals"
            # print("vocals_dir", vocals_dir) # folder with the actual files to transcribe
            # move_vocals_files(vocals_dir)
            print("separate_vocals not implemented fully")
        else:
            vocals_dir = args.audio_dir

        print("Model: ", args.model)

        # Transcribe each audio vocal file in vocals_dir. filename is base filename, ex test01.wav. case_name is without .wav, ex. test01. audio_fp is whole path to audio
        for filename in os.listdir(vocals_dir):
            if filename == ".DS_Store": continue

            case_name = str(filename.split(".")[0])
            if case_name in transcribed_filenames:
                print(f"Skipping {case_name}, already transcribed")
                continue
            
            # Transcribe vocals
            audio_fp = os.path.join(vocals_dir, filename)
            print(f"Transcribing {audio_fp} with use_prompt={args.use_prompt}; saving to {args.save_dir}/{args.save_filename}")
            if args.use_prompt:
                transcribed_df.loc[len(transcribed_df)] = [case_name, "", transcribe(audio_fp, args.model, prompt=prompt)]
            else:
                transcribed_df.loc[len(transcribed_df)] = [case_name, "", transcribe(audio_fp, args.model)]

            
            transcribed_df.to_csv(output_fp, index=False) # Save intermediate results
            print(f"Transcribed {case_name}, saved to {output_fp}")

        print(f"Saved transcriptions to {args.save_dir}")

        


    ############# Evaluation - !outdated ###################
    if do_evaluate_results:
        print("Evaluation of resulting transcript:")
        processed_pred_df = process_predictions(output_fp)
        wer_whisper = compute_metrics(processed_pred_df, 
                                    labels=args.test_labels_fp)
        print(f"WER for {args.model}: {wer_whisper}")




'''
## with first training + eval datasets:
    finetuned, 15 epochs /scratch/eguan2/whisper {34.535367545076284}  
    WER for distil-whisper/distil-large-v3: {'wer': 26.0748959778086}
    WER for distil-whisper/distil-large-v3: {'wer': 34.11927877947296} <- second time. idk why 1st was so good
    WER for /scratch/eguan2/distil_5: {'wer': 35.367545076282944} (deleted this checkpoint from scratch since)
    WER for openai/whisper-large-v3 (not finetuned): {'wer': 31.206657420249652}
    WER for /scratch/eguan2/lg_50: {'wer': 31.345353675450767}

## with slightly larger sets (12/5):

    DEBUG WHY IT DIDN'T RESUME FROM CHECKPOINT
{'eval_loss': 0.711867094039917, 'eval_wer': 15.430861723446892, 'eval_runtime': 17.973, 'eval_samples_per_second': 0.89, 'eval_steps_per_second': 0.167, 'epoch': 6.0}     
{'eval_loss': 0.528700053691864, 'eval_wer': 11.623246492985972, 'eval_runtime': 18.1883, 'eval_samples_per_second': 0.88, 'eval_steps_per_second': 0.165, 'epoch': 18.0}                
{'eval_loss': 0.459404319524765, 'eval_wer': 9.318637274549097, 'eval_runtime': 18.5896, 'eval_samples_per_second': 0.861, 'eval_steps_per_second': 0.161, 'epoch': 24.0}             
{'eval_loss': 0.33771249651908875, 'eval_wer': 12.925851703406813, 'eval_runtime': 18.2966, 'eval_samples_per_second': 0.874, 'eval_steps_per_second': 0.164, 'epoch': 36.0}
{'eval_loss': 0.3075251579284668, 'eval_wer': 16.132264529058116, 'eval_runtime': 17.9126, 'eval_samples_per_second': 0.893, 'eval_steps_per_second': 0.167, 'epoch': 40.0} 
{'eval_loss': 0.2902292311191559, 'eval_wer': 15.631262525050099, 'eval_runtime': 18.6017, 'eval_samples_per_second': 0.86, 'eval_steps_per_second': 0.161, 'epoch': 44.0}  
{'eval_loss': 0.2885704040527344, 'eval_wer': 15.631262525050099, 'eval_runtime': 18.2702, 'eval_samples_per_second': 0.876, 'eval_steps_per_second': 0.164, 'epoch': 45.0}
{'train_runtime': 1801.5761, 'train_samples_per_second': 1.948, 'train_steps_per_second': 0.05, 'train_loss': 1.1068094889322917, 'epoch': 45.0}

EVAL:
    WER for /scratch/eguan2/whisper_lg_45: {'wer': 31.071428571428573}
    WER for openai/whisper-large-v3: {'wer': 31.30952380952381}

eval with 72 (overlaps with training data, but just to test): 
    WER for /scratch/eguan2/whisper_lg_45: {'wer': 30.753768844221106}
    WER for openai/whisper-large-v3: {'wer': 31.758793969849247}
'''
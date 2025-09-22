###################### https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
import pandas as pd
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
from .utils import process_predictions, compute_metrics, copy_process_streams, separate_vocals, move_vocals_files



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


if __name__ == "__main__":
    '''
    Run from `endoscribe` folder root, e.g.:
    
    python transcription/whisper_transcribe.py \
    --procedure_type=col --save_filename=whisper_lg_v3 \
    --model=/scratch/eguan2/whisper_lg_45 \
    --audio_dir=transcription/recordings/finetune_testset

    EUS (cyst):
    python transcription/whisper_transcribe.py \
    --procedure_type=eus --save_filename=whisper_lg_v3_clean_noprompt \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/eus/cyst

    ERCP
    python transcription/whisper_transcribe.py \
    --procedure_type=ercp --save_filename=whisper_lg_v3 \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/ercp/bdstone

    EGD
    python transcription/whisper_transcribe.py \
    --procedure_type=egd --save_filename=whisper_lg_v3 \
    --model=openai/whisper-large-v3 \
    --audio_dir=transcription/recordings/egd

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='Path to audio folder (general, not specialized vocals folder)')
    parser.add_argument('--procedure_type', choices=['col', 'eus', 'ercp', 'egd'], required=True, help="Type of procedure to process")
    parser.add_argument('--save_filename', type=str, required=True, help='Name of file, without .csv, to save transcriptions to')
    parser.add_argument('--do_separate_vocals', type=str,  default=False, help='Whether to run DEMUCS vocal separation on audio recordings')
    parser.add_argument('--model', type=str, required=True) #'distil-whisper/distil-large-v3' #openai/whisper-medium.en #openai/whisper-large-v3') 
    parser.add_argument('--use_prompt', default=False)
    parser.add_argument('--test_labels_fp', 
                            default='transcription/finetune_data/test_labels.csv') # CSV of gold transcripts, columns=[file, transcript]
    args = parser.parse_args()

    save_dir = f"transcription/results/{args.procedure_type}"
    output_fp = f"{save_dir}/{args.save_filename}.csv"

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

        #!outdated. Separate vocals or not
        if args.do_separate_vocals: 
            # separate_vocals(args.audio_dir, output_audio_path=args.audio_dir) # creates a subfolder `htdemucs` in the output path
            # vocals_dir = f"{args.audio_dir}/vocals"
            # print("vocals_dir", vocals_dir) # folder with the actual files to transcribe
            # move_vocals_files(vocals_dir, args.audio_dir)
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
            print(f"Transcribing {audio_fp} with use_prompt={args.use_prompt}; saving to {save_dir}/{args.save_filename}")
            if args.use_prompt:
                transcribed_df.loc[len(transcribed_df)] = [case_name, "", transcribe(audio_fp, args.model, prompt=prompt)]
            else:
                transcribed_df.loc[len(transcribed_df)] = [case_name, "", transcribe(audio_fp, args.model)]

            
            transcribed_df.to_csv(output_fp, index=False) # Save intermediate results
            print(f"Transcribed {case_name}, saved to {output_fp}")

        print(f"Saved transcriptions to {save_dir}")

        


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
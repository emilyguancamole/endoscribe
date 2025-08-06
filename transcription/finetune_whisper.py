import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import sys
import logging
import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

''' 
Run from endoscribe/transcription:
    torchrun --nproc_per_node=3 finetune_whisper.py --train_strategy epoch --num_epochs=45 --train_datasets finetune_data/train --eval_datasets finetune_data/eval --output_dir /scratch/eguan2/whisper_lg_45

Requirements: transformers accelerate evaluate jiwer tensorboard gradio

Code is based on:
https://huggingface.co/blog/fine-tune-whisper 
https://github.com/vasistalodagala/whisper-finetune/blob/master/train/fine-tune_on_custom_dataset.py

'''


#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument('--train_strategy', type=str, required=False, 
    default='steps', help='Training strategy: steps or epoch.'
)
parser.add_argument('--learning_rate', type=float, required=False, 
    default=5e-6, help='Learning rate for fine-tuning.'
)
parser.add_argument('--warmup', type=int, required=False, 
    default=250, help='Number of warmup steps.'
) # repo had set to 20k
parser.add_argument('--train_batchsize', type=int, required=False, 
    default=16, help='Batch size during the training phase.'
)
parser.add_argument('--eval_batchsize', type=int, required=False, 
    default=2, help='Batch size during the evaluation phase.'
)
parser.add_argument('--num_epochs', type=int, required=False, 
    default=20, help='Number of epochs to train for.'
)
parser.add_argument('--num_steps', type=int, required=False, 
    default=100000, help='Number of steps to train for.'
)
parser.add_argument('--resume_from_ckpt', type=str, required=False, 
    default=None, help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument('--output_dir', type=str, 
    required=True, help='Output directory for the checkpoints generated.'
)
parser.add_argument('--train_datasets', type=str, nargs='+', 
    required=True, default=[], help='List of datasets to be used for training.'
)
parser.add_argument('--eval_datasets', type=str, nargs='+', 
    required=True, default=[], help='List of datasets to be used for evaluation.'
)
args = parser.parse_args()
if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')


# Redirect logs to both console and file
log_filename = f"{args.output_dir}/training_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode="w"),  # save logs to file
        logging.StreamHandler(sys.stdout)  # print logs to console
    ]
)


########################       MODEL LOADING       ##########################
model_name = 'openai/whisper-large-v3'#distil-whisper/distil-large-v3'
language = "English"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

#############       MY DATASETS      #############
# train_datasets = ['data_finetune/output_train_data']
# eval_datasets = ['data_finetune/output_eval_data']
def load_custom_dataset(split):
    ds = []
    if split == 'train':
        for dset in args.train_datasets:
            # ds.append(dset)
            ds.append(load_from_disk(dset))
    if split == 'eval':
        for dset in args.eval_datasets:
            # ds.append(dset)
            ds.append(load_from_disk(dset))

    ds_to_return = concatenate_datasets(ds)
    ds_to_return = ds_to_return.shuffle(seed=22)
    return ds_to_return

def prepare_dataset(batch):
    ''' Prepare a batch of data '''
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    # optional pre-processing steps
    transcription = batch["sentence"]
    # if do_lower_case:
    #     transcription = transcription.lower()
    # if do_remove_punctuation:
    #     transcription = normalizer(transcription).strip()

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


''' Prepare all our data for training '''
max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length

sampling_rate = 16000
num_proc = 1 # needs to be 1 on IA1

print('DATASET PREPARATION IN PROGRESS...')
raw_dataset = DatasetDict()
raw_dataset["train"] = load_custom_dataset('train')
raw_dataset["eval"] = load_custom_dataset('eval')

raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate)) # set the audio inputs to the correct sampling rate, done on the fly
raw_dataset = raw_dataset.map(prepare_dataset, num_proc=num_proc) # apply the data preparation function to all of our training examples

raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["input_length", "labels"],
    num_proc=num_proc,
)



####### Define a Data Collator https://huggingface.co/blog/fine-tune-whisper#define-a-data-collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
       
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # labels_batch = self.processor.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # print("LABELS", labels) # prints a long tensor

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print('DATASET PREPPED')


########   DEFINE EVAL METRICS   #######
print("Prepping eval metrics")
def compute_metrics(pred):
    metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    loss = trainer.state.log_history[-1].get("loss", None)  # Get latest loss
    logging.info(f"Training loss: {loss}") #? save loss to file

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}



#########   TRAINING ARGS AND TRAINING   ###########


do_normalize_eval = False
gradient_checkpointing = True

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=5,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=1000,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )


# forward the training arguments to the 🤗 Trainer along with our model, dataset, data collator and compute_metrics function
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # tokenizer=processor #.feature_extractor,
    processing_class=processor
)

processor.save_pretrained(training_args.output_dir) # Processor is not trainable; won't change

# To launch training:
print("LAUNCHING TRAINING")
trainer.train()
print("TRAINING COMPLETE")

trainer.save_model(args.output_dir)  # saves model weights and config files

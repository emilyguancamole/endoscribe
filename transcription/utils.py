import os
from transcription.convert_to_mono import batch_convert
from word2number import w2n
import roman
import contractions
import re
import pandas as pd
import evaluate

import subprocess as sp
import io
from typing import Dict, Tuple, Optional, IO
from pathlib import Path
import sys
import shutil
import select

def process_predictions(pred_file, transcript_col_name="pred_transcript"):
    ''' Process predictions to be lowercase, remove punctuation, expand contractions, make numbers digits.
        For evaluation of WER only.
    '''
    pred_df = pd.read_csv(pred_file)
    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: x.lower()) # lowercase
    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: contractions.fix(x)) # contractions
    # pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: x.strip("[]()'")) # especially parakeet seems to keep these characters as part of the transciption
    # Convert numbers to digits
    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: numbers_to_digits(x))

    pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: re.sub(r"[^\w\s]", "", str(x))) # punctuation
    # pred_df[transcript_col_name] = pred_df[transcript_col_name].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # for x in pred_df[transcript_col_name]: print(x)
    
    return pred_df


def numbers_to_digits(input_string):
    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90, "hundred": 100
    }

    words = input_string.split()
    result = [] 
    current_number = 0 
    for word in words:
        if word in number_words:
            if word=="hundred" or word=="thousand":
                if current_number == 0:
                    current_number = 1  # Handle cases like one hundred by multiplying by scale
                current_number *= number_words[word] 
            else:
                current_number += number_words[word]  # Add value of number word
        # For some reason parakeet randomly outputs some numbers in roman numerals, so check for those
        elif word!='i' and all(char in "ivxlcdm" for char in word):  # If all characters in word are Roman chars #! would there ever be an individual x that isn't a number?
            try:
                roman_value = roman.fromRoman(word.upper())
                current_number += roman_value
            except roman.InvalidRomanNumeralError:
                result.append(word)  # Append unrecognized Roman numeral
        else:
            # Reach a word that isn't a number, so finalize current number
            if current_number > 0:
                result.append(str(current_number))
                current_number = 0  
            result.append(word)
    if current_number > 0:
        result.append(str(current_number))

    return ' '.join(result)

def compute_metrics(predictions, labels):
    ''' Compute average WER (Word Error Rate) across all samples.
    Adapted from: https://huggingface.co/blog/fine-tune-whisper#training-and-evaluation
    pred_file: path to csv file with transcriptions, OR dataframe
    label_file: path to csv file with ground truth transcriptions OR dataframe
    '''
    metric = evaluate.load("wer")
    # if pandas df, use directly, else read csv
    pred_df = predictions if isinstance(predictions, pd.DataFrame) else pd.read_csv(predictions)

    label_df = labels if isinstance(labels, pd.DataFrame) else pd.read_csv(labels)
    
    merged_df = pred_df.merge(label_df, on='participant_id', how='left')
    wer = 100 * metric.compute(predictions=merged_df["pred_transcript"], references=merged_df["transcript"])
    return {"wer": wer}

####### VOCAL PROCESSING - not used currently 9/14 ##########
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
    
    files = [f for f in Path(input_audio_path).iterdir() if f.is_file()]

    print("Files to separate:", files)
    print("With command: ", " ".join(cmd))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")

def move_vocals_files(vocals_dir, audio_dir):
    # Move vocals files to a new location vocals/{filename}_vocals -- transcription should use this
    htdemucs_dir = os.path.join(audio_dir, "htdemucs")
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

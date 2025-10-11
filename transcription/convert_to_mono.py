import argparse
from pydub import AudioSegment
import os

def convert_file_mono_wav(audiofile, save_fp, audio_format, replace=False):
    '''
    Process a audio file to be mono channel, 16 kHz, wav format.
            pydub uses ffmpeg.
    Args:
        audiofile: path (filename) of audiofile to convert. 
        save_fp: path (filename) to save the converted audiofile.
    '''
    
    audio = AudioSegment.from_file(audiofile, format=audio_format)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(save_fp, format="wav")
    if replace:
        print(audiofile)
        os.remove(audiofile)
    
        

def batch_convert(audio_dir, inp_audio_format, replace=False):
    '''
    Convert all audio files in a directory to wav mono.
    Args:
        audio_dir: directory containing audio files.
        save_dir: directory to save converted files.
        inp_audio_format: format of the input audio files ('wav', 'm4a').
    '''
    save_dir = audio_dir # if replace else os.path.join(audio_dir, "converted")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for audiofile in os.listdir(audio_dir):
        if audiofile.startswith('.') or audiofile.endswith(".wav") or os.path.isdir(audiofile):
            continue
        input_path = os.path.join(audio_dir, audiofile)
        output_filename = os.path.splitext(audiofile)[0] + ".wav"
        print(output_filename)
        print("output", output_filename)
        save_fp = os.path.join(save_dir, output_filename)
        convert_file_mono_wav(input_path, save_fp, inp_audio_format, replace)

    print(f"Converted files saved in {save_dir}")


if __name__ == "__main__":
    ''' Run as standalone script
    python transcription/convert_to_mono.py --audio_format=m4a --audio_dir=recordings/eus/cyst --save_dir=recordings/eus/cyst_mono
    '''

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--audio_format', type=str, required=True, choices=['wav', 'm4a'])
    argparser.add_argument('--audio_dir', type=str, required=True)
    argparser.add_argument('--save_dir', type=str, required=True)
    argparser.add_argument('--replace', default=False)
    args = argparser.parse_args()

    # audio_dir = "/Users/emilyguan/Downloads/EndoScribe/recordings/eus/cancer"
    # save_dir = "/Users/emilyguan/Downloads/EndoScribe/recordings/eus/cancer_mono"

    for audiofile in os.listdir(args.audio_dir):
        if audiofile.startswith('.'):
            continue
        output_filename = audiofile.split(".")[0] + ".wav"
        convert_file_mono_wav(f"{args.audio_dir}/{audiofile}", f"{args.save_dir}/{output_filename}", args.audio_format, args.replace)
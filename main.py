

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
import argparse
import pandas as pd
from llm.llm_client import LLMClient
from processors import ColProcessor, ERCPProcessor, EUSProcessor, EGDProcessor
import torch
import argparse

def main():
    ''' 
    python main.py --procedure_type=col --transcripts_fp=first_datasets/abstract_transcribe_fall24.csv --output_filename=082025-test --to_postgres --files_to_process 16 11

    python main.py --procedure_type=eus --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process cancer01 mass01

    python main.py --procedure_type=ercp --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process bdstone01 bdstricture01

    python main.py --procedure_type=egd --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process egd01 egd02 egd03 egd04 egd05
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--procedure_type', choices=['col', 'eus', 'ercp', 'egd'], required=True, help="Type of procedure to process")
    parser.add_argument('--transcripts_fp', required=True, help="Relative path to transcripts CSV file within transcription/results/{args.procedure_type} folder")
    parser.add_argument('--output_filename', required=True, help="File name to save the extracted outputs. Will be saved as a .csv in ./results/{args.procedure_type}")
    parser.add_argument('--to_postgres', action='store_true', help="If set, write extracted outputs directly to Postgres") # TODO
    parser.add_argument('--files_to_process', nargs='*', help="List of filenames to process; 'all' to process all files")
    args = parser.parse_args()

    system_prompt_fp = f"prompts/{args.procedure_type}/system.txt" #!
    # args.model_dir = "/scratch/eguan2/llama33-70/llama33-70_model"
    # tok_dir = "/scratch/eguan2/llama33-70/llama33-70_tokenizer"
    output_fp = f"./results/{args.procedure_type}/{args.output_filename}.csv"

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device no.{i}: {torch.cuda.get_device_name(i)}")

    llm_handler = LLMClient(
        model_path="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",#"Qwen/Qwen3-14B"
        quant="compressed-tensors",
        tensor_parallel_size=4,
        # Llama 3.3 #"meta-llama/Llama-3.3-70B-Instruct", # vllm loads from cache, /scratch/eguan2/hf_cache/hub
    )

    # Map procedure type to processor class and transcript path
    processor_map = {
        "col": (ColProcessor, f"transcription/results/col/{args.transcripts_fp}"),
        "eus": (EUSProcessor, f"transcription/results/eus/{args.transcripts_fp}"),
        "ercp": (ERCPProcessor, f"transcription/results/ercp/{args.transcripts_fp}"),
        "egd": (EGDProcessor, f"transcription/results/egd/{args.transcripts_fp}"),
    }

    processor_class, transcript_path = processor_map[args.procedure_type]
    transcripts_df = pd.read_csv(transcript_path)
    transcripts_df["file"] = transcripts_df["file"].astype(str)
    processor = processor_class(args.procedure_type, system_prompt_fp, output_fp, llm_handler, args.to_postgres)
    processor.process_transcripts(args.files_to_process, transcripts_df)

    print(f"Processing complete for {args.procedure_type}. Outputs saved to {output_fp}")


if __name__=="__main__":
    main()

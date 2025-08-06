

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,9"
import argparse
import pandas as pd
from llm.llm_handler import LLMHandler
from processors.col_processor import ColProcessor
from processors.ercp_processor import ERCPProcessor
from processors.eus_processor import EUSProcessor
import torch
import argparse

def main():
    ''' 
    python main.py --procedure_type=col --transcripts_fp=first_datasets/abstract_transcribe_fall24.csv --output_filename=082025-test --to_postgres --files_to_process 16 11

    python main.py --procedure_type=eus --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process cancer01 mass01

    python main.py --procedure_type=ercp --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process bdstone01 bdstricture01
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--procedure_type', choices=['col', 'eus', 'ercp'], required=True, help="Type of procedure to process")
    parser.add_argument('--transcripts_fp', required=True, help="Relative path to transcripts CSV file within transcription/{args.procedure_type}_results folder")
    parser.add_argument('--output_filename', required=True, help="File name to save the extracted outputs, e.g. '1_outputs'. Will be saved as a .csv in ./results/{args.procedure_type}")
    parser.add_argument('--to_postgres', action='store_true', help="If set, write extracted outputs directly to Postgres")
    parser.add_argument('--files_to_process', nargs='*', help="List of filenames to process; 'all' to process all files")
    args = parser.parse_args()

    system_prompt_fp = f"05_{args.procedure_type}_experiments/prompts/system.txt"
    # args.model_dir = "/scratch/eguan2/llama33-70/llama33-70_model"
    # tok_dir = "/scratch/eguan2/llama33-70/llama33-70_tokenizer"
    output_fp = f"./results/{args.procedure_type}/{args.output_filename}.csv"

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device no.{i}: {torch.cuda.get_device_name(i)}")

    llm_handler = LLMHandler(
        model_path="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
        quant="compressed-tensors",
        tensor_parallel_size=4,
        # model_path="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", # Llama 3.3 #"meta-llama/Llama-3.3-70B-Instruct", # vllm should load from cache since model's alr downloaded args.model_dir
        # quant="awq_marlin", # "Use quant=awq_marlin for faster inference"
    )

    transcripts_df = pd.read_csv(f"transcription/{args.procedure_type}_results/{args.transcripts_fp}")

    # Map procedure type to processor class and transcript path
    processor_map = {
        "col": (ColProcessor, f"transcription/col_results/{args.transcripts_fp}"),
        "eus": (EUSProcessor, f"transcription/eus_results/{args.transcripts_fp}"),
        "ercp": (ERCPProcessor, f"transcription/ercp_results/{args.transcripts_fp}"),
    }

    if args.procedure_type not in processor_map:
        raise ValueError(f"Unknown procedure type: {args.procedure_type}")

    processor_class, transcript_path = processor_map[args.procedure_type]
    transcripts_df = pd.read_csv(transcript_path)
    transcripts_df["file"] = transcripts_df["file"].astype(str)
    processor = processor_class(args.procedure_type, system_prompt_fp, output_fp, llm_handler, args.to_postgres)
    processor.process_transcripts(args.files_to_process, transcripts_df)

    print(f"Processing complete for {args.procedure_type}. Outputs saved to {output_fp}")


if __name__=="__main__":
    main()

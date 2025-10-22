

import os
import argparse
import pandas as pd
import torch

# Dynamic CUDA configuration - only set if CUDA is available
if torch.cuda.is_available():
    # Use CUDA_VISIBLE_DEVICES env var if set, otherwise use default GPUs
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6,9"
        print(f"CUDA detected. Setting CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("CUDA not available. Running on CPU or MPS (Apple Silicon).")

from llm.llm_client import LLMClient
from processors import ColProcessor, ERCPProcessor, EUSProcessor, EGDProcessor

def infer_procedure_type(row):
        for proc in ['egd', 'eus', 'ercp', 'colonoscopy', 'endoflip', 'sigmoidoscopy']:
            if row[f'procedure_type_{proc}'] == 1:
                return proc if proc != 'colonoscopy' else 'col'
        return row['procedure_type_other'].lower() if pd.notna(row['procedure_type_other']) else 'unknown'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--procedure_type', choices=['col', 'eus', 'ercp', 'egd'], required=True, help="Type of procedure to process")
    parser.add_argument('--transcripts_fp', required=True, help="Relative path to transcripts csv file within transcription/results/{args.procedure_type} folder")
    parser.add_argument('--output_filename', required=True, help="File name to save the extracted outputs. Will be saved as a .csv in ./results/{args.procedure_type}")
    parser.add_argument('--to_postgres', action='store_true', help="If set, write extracted outputs directly to Postgres") # TODO
    parser.add_argument('--files_to_process', nargs='*', help="List of filenames to process; 'all' to process all files")
    
    # Model config options
    parser.add_argument('--model_config', choices=['local_llama', 'openai_gpt4o', 'anthropic_claude'],
                       default='local_llama', help="Predefined model configuration to use")
    parser.add_argument('--model_type', choices=['local', 'openai', 'anthropic'], help="Override model type (local, openai, or anthropic)")
    parser.add_argument('--model_path', help="Override model path, OpenAI model name, or Anthropic model name")
    
    args = parser.parse_args()

    system_prompt_fp = f"prompts/{args.procedure_type}/system.txt"
    # args.model_dir = "/scratch/eguan2/llama33-70/llama33-70_model"
    output_fp = f"results/{args.procedure_type}/{args.output_filename}.csv"

    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device no.{i}: {torch.cuda.get_device_name(i)}")

    # Initialize LLM with flexible configuration
    llm_kwargs = {}
    if args.model_type: llm_kwargs['model_type'] = args.model_type
    if args.model_path: llm_kwargs['model_path'] = args.model_path
    
    # Use predefined config or custom parameters
    if args.model_config and not any([args.model_type, args.model_path]):
        llm_handler = LLMClient.from_config(args.model_config, **llm_kwargs) #* `from_config` classmethod instead of passing in params to LLMClient constructor
        print(f"Using predefined model config: {args.model_config}")
    else:
        # Fallback to pre-10/5/2025 local model config
        llm_handler = LLMClient(
            model_path=args.model_path or "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
            model_type=args.model_type or "local",
            quant="compressed-tensors" if args.model_type == "local" else None,
            tensor_parallel_size=4 if args.model_type == "local" else None,
            **llm_kwargs
        )
        print(f"Using custom model config: {args.model_type or 'local'} - {args.model_path or 'RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16'}")

    ## Map procedure type to processor class and transcript path #* old - changed to transcripts for all procedure types in `long` folder
    processor_map = {
        "col": ColProcessor,
        "eus": EUSProcessor,
        "ercp": ERCPProcessor,
        "egd": EGDProcessor,
    }

    processor_class = processor_map[args.procedure_type]
    transcripts_df = pd.read_csv(f"transcription/results/{args.procedure_type}/{args.transcripts_fp}")
    transcripts_df["participant_id"] = transcripts_df["participant_id"].astype(str)
    processor = processor_class(args.procedure_type, system_prompt_fp, output_fp, llm_handler, args.to_postgres)
    processor.process_transcripts(args.files_to_process, transcripts_df)

    print(f"Processing complete for {args.procedure_type}. Outputs saved to {output_fp}")


if __name__=="__main__":
    ''' 
    python main.py --procedure_type=col --transcripts_fp=long-10-2025.csv --output_filename=longform/long-10-2025 --files_to_process all

    python main.py --procedure_type=eus --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process cancer01 mass01

    python main.py --procedure_type=ercp --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process bdstone01 bdstricture01

    python main.py --procedure_type=egd --transcripts_fp=whisper_lg_v3.csv --output_filename=082025-test --files_to_process egd01 egd02 egd03 egd04 egd05

    LONGFORM, OPENAI FROM CONFIG
        python main.py --procedure_type=egd --transcripts_fp=long-10-2025.csv --output_filename=longform/long-10-2025 --files_to_process all --model_config=openai_gpt4o
    '''

    main()

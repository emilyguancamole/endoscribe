import os
import tempfile

from dotenv import load_dotenv
from openai import AzureOpenAI
import gradio as gr
import pandas as pd
from docx import Document
from transcription.whisper_transcribe import transcribe as local_transcribe
from processors.col_processor import ColProcessor
from drafters.colonoscopy import ColonoscopyDrafter
from llm.llm_client import LLMClient



# Map UI names to huggingface model ids for Whisper
WHISPER_MODEL_OPTIONS = {
    "Whisper Base": "openai/whisper-base",
    "Whisper Small": "openai/whisper-small",
    "Whisper Medium": "openai/whisper-medium",
    'Turbo': "openai/whisper-large-v3-turbo",
    "Distil": "distil-whisper/distil-large-v3",
    "Qwen 3-14B": "Qwen/Qwen3-14B"
}

# Set up GPT
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# client = AzureOpenAI(
#     azure_endpoint="https://gpt4-endoscribe.openai.azure.com/",
#     api_version="2025-02-01-preview",
#     api_key=api_key
# )
# print(f"Using Gpt")

# Set up LLM (Llama 4, quantized, as in main.py)
llm_handler = LLMClient(
    model_path="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
    quant="compressed-tensors",
    tensor_parallel_size=4,
)


def transcribe(audio_file, whisper_model_name):
    # Use local pipeline, but allow model override
    model_id = WHISPER_MODEL_OPTIONS[whisper_model_name]
    # local_transcribe expects (audio_file, whisper_model, prompt=None)
    return local_transcribe(audio_file, model_id)


def full_pipeline(audio_path, whisper_model_name, sample_id):
    # 1. Transcription
    transcription = transcribe(audio_path, whisper_model_name)

    # 2. Extraction using ColProcessor and LLMClient
    # Build a fake transcripts_df for a single sample
    transcripts_df = pd.DataFrame({
        "file": [sample_id],
        "pred_transcript": [transcription],
    })

    system_prompt_fp = "prompts/col/system.txt"
    output_fp = tempfile.mktemp(suffix=".csv")
    processor = ColProcessor(
        procedure_type="col",
        system_prompt_fp=system_prompt_fp,
        output_fp=output_fp,
        llm_handler=None,  # using Azure OpenAI directly in ColProcessor
        to_postgres=False,
    )

    # Only process this sample
    processor.process_transcripts([sample_id], transcripts_df)

    # Read the output CSVs
    colon_fp = output_fp.replace(".csv", "_colonoscopies.csv")
    polyp_fp = output_fp.replace(".csv", "_polyps.csv")
    pred_df = pd.read_csv(colon_fp, index_col="id")
    polyp_df = pd.read_csv(polyp_fp) if os.path.exists(polyp_fp) else None

    # 3. Generate report using ColonoscopyDrafter
    drafter = ColonoscopyDrafter(sample_id, pred_df, polyp_df)
    doc = drafter.draft_doc()

    # 4. Save and preview report (optional)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        report_path = tmp.name

    return transcription, report_path


# Gradio interface
interface = gr.Interface(
    fn=full_pipeline,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(choices=list(WHISPER_MODEL_OPTIONS.keys()), value="Whisper Base", label="Select Whisper Model"),
        gr.Textbox(label="Sample ID", placeholder="Case/test/sample number"),
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.File(label="Generated Report (.docx)")
    ],
    title="Endoscopy Transcription & Report Generator",
    description="Upload an audio file to generate a structured endoscopy report using Whisper and Llama 4."
)

interface.launch(share=True)
import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import gradio as gr
import torch
from transformers import pipeline
from gpt_extraction import process_transcript
from drafters.origdrafter import create_colon_report_docx
import tempfile


WHISPER_MODEL_OPTIONS = {
    "Whisper Base": "openai/whisper-base",
    "Whisper Small": "openai/whisper-small",
    "Whisper Medium": "openai/whisper-medium",
    # "MedAI": "Crystalcareai/Whisper-Medicalv1", # distil-based, no good
    'Turbo': "openai/whisper-large-v3-turbo",
    "Distil": "distil-whisper/distil-large-v3",
}

# Set up GPT
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AzureOpenAI(
    azure_endpoint="https://gpt4-endoscribe.openai.azure.com/",
    api_version="2025-02-01-preview",
    api_key=api_key
)
print(f"Using Gpt")


def transcribe(audio_file, whisper_model_name):
    model_id = WHISPER_MODEL_OPTIONS[whisper_model_name]
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("automatic-speech-recognition", model=model_id, device=device)
    result = pipe(audio_file)
    return result["text"]


def convert_json_to_df(json_output, sample_id):
    """ Converts GPT JSON output to a dataframe for report generation. """
    import pandas as pd
    df = pd.DataFrame([json_output], index=[sample_id])
    return df

def construct_prompt(transcript_to_process):
    ''' based on gpt_extraction.py/construct_prompt, but hardcoded prompt construction '''
    prompt_field_def_fp = 'extraction_prompts/colonoscopy/field_definitions_v3.txt'
    prompt_fp = 'extraction_prompts/colonoscopy/prompt_base.txt'
    examples_fp = 'extraction_prompts/colonoscopy/examples_prompt.txt'
    prompt_fields_def = open(prompt_field_def_fp).read()
    prompt_base = open(prompt_fp).read()
    examples = open(examples_fp).read()
    prompt = prompt_base.replace('{{prompt_field_definitions}}', prompt_fields_def).\
            replace('{{examples_prompt}}', examples).\
            replace('{{transcript_to_process}}', transcript_to_process)
    return prompt

def full_pipeline(audio_path, whisper_model_name, sample_id):
    # 1. Transcription
    transcription = transcribe(audio_path, whisper_model_name)
    
    # 2. GPT Extraction
    prompt = construct_prompt(transcription)
    print("Prompt for GPT\n", prompt)
    # extracted_json = process_transcript(prompt)  # returns JSON string
    cur_res = process_transcript(prompt, 'gpt-4o', client=client)
    print("RAW Extract Result\n", cur_res)
    
    # For formatting as nice json, first process output into a dictionary
    start = cur_res.find("{")
    end = cur_res.rfind("}") + 1
    if start == -1 or end == 0:
        print(f"No JSON found, skipping.")
    cur_res = cur_res[start:end].strip()
    try: # Try parsing JSON
        cur_res_dict = json.loads(cur_res)
        print(f"Parsed JSON successfully.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}") # Print to debug
        print(f"Cleaned JSON attempt: {cur_res}")

    # 3. Generate report (could also return markdown string if not saving .docx)
    pred_df = convert_json_to_df(cur_res_dict, sample_id=sample_id)
    doc = create_colon_report_docx(sample_id, pred_df)

    # 4. Save and preview report (optional)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        report_path = tmp.name

    return transcription, report_path #, cur_res_dict

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
        # gr.JSON(label="Extracted Entities (via GPT)"), // json output
        gr.File(label="Generated Report (.docx)")
    ],
    title="Endoscopy Transcription & Report Generator",
    description="Upload an audio file to generate a structured endoscopy report using Whisper and GPT."
)

# Launch app
interface.launch(share=True)
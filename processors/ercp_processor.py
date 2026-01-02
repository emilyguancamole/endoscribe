import json
import os
from .base_processor import BaseProcessor
import pandas as pd
from models.generated_ercp_base_model import ErcpBaseData
from models.data_models import PEPRiskData

# NOTE 1/1/2026: DEPRECATED

class ERCPProcessor(BaseProcessor):
    def extract_pep_from_transcript(self, transcript: str, filename: str = "live") -> dict:
        """
        Run PEP risk extraction on a single transcript and return a validated dict.
        This is a lightweight wrapper for server-side, one-off processing.
        Returns a dict with keys: id, model, and the ERCPData fields.
        """
        # Use absolute paths to avoid working directory issues
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_field_definitions_fp = os.path.join(base_dir, 'pep_risk', 'prompts', 'field_definitions.txt')
        fewshot_examples_dir = os.path.join(base_dir, 'pep_risk', 'prompts', 'fewshot')

        messages = self.build_messages(
            transcript,
            prompt_field_definitions_fp=prompt_field_definitions_fp,
            fewshot_examples_dir=fewshot_examples_dir,
            prefix="pep",
        )

        if self.llm_handler.model_type == "local":
            response = self.llm_handler.chat(messages)[0].outputs[0].text.strip()
        elif self.llm_handler.model_type in ["openai", "anthropic"]:
            response = self.llm_handler.chat(messages)

        json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
        # print("PEP extraction raw json result:\n", json_response)
        validated = PEPRiskData(**json_response)
        print("PEP extraction validated result:\n", validated)
        return {
            "id": filename,
            "model": self.llm_handler.model_type,
            **validated.dict(),
        }
    
    def process_transcripts(self, filenames_to_process, transcripts_df):
        print(f"Processing ERCP transcripts...")
        outputs = []
        prompt_field_definitions_fp = './prompts/ercp/generated_ercp_base_prompt.txt'
        for _, row in transcripts_df.iterrows():
            if filenames_to_process[0] != "all" and (row["participant_id"] not in filenames_to_process):
                continue

            cur_transcript = row["pred_transcript"]
            filename = row["participant_id"]

            print(f"File: {filename} - Transcript: {cur_transcript[:200]}")
            messages = self.build_messages(
                cur_transcript,
                prompt_field_definitions_fp=prompt_field_definitions_fp,  
                # fewshot_examples_dir=None,
                # prefix="ercp" #! todo fewshot examples
            )
            if self.llm_handler.model_type == "local": #! diff response processing for local vs openai
                pass
            elif self.llm_handler.model_type in ["openai", "anthropic"]:
                response = self.llm_handler.chat(messages)
            try:
                json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
                validated = ErcpBaseData(**json_response)
            except json.JSONDecodeError:
                continue
            
            # Parse ERCP response - not in a sep file (as in col_processor)
            outputs.append({
                "id": filename,
                "model": self.llm_handler.model_type,
                **validated.dict()
            })

        # Save outputs to csv
        self.save_outputs(outputs)

    def save_outputs(self, outputs):
        ercp_df = pd.DataFrame(outputs)
        # Save CSV (append) 
        self.save_dataframe(ercp_df, self.output_fp, index=False)
        # Upsert to Postgres if requested
        self.upsert_dataframe(ercp_df, "ercp_procedures")
import json
from pydantic import ValidationError
from .base_processor import BaseProcessor
import pandas as pd
from data_models.data_models import EUSData

class EUSProcessor(BaseProcessor):
    def process_transcripts(self, filenames_to_process, transcripts_df):
        outputs = []
        #! Prompt files for EUS
        prompt_field_definitions_fp = './prompts/eus/eus.txt'
        fewshot_examples_dir = "./prompts/eus/fewshot"

        for _, row in transcripts_df.iterrows():
            if filenames_to_process[0] != "all" and row["participant_id"] not in filenames_to_process:
                continue

            cur_transcript = row["pred_transcript"]
            filename = row["participant_id"]

            messages = self.build_messages(
                cur_transcript,
                prompt_field_definitions_fp=prompt_field_definitions_fp,
                fewshot_examples_dir=fewshot_examples_dir,
                prefix="eus"
            )

            if self.llm_handler.model_type == "local": #! diff response processing for local vs openai
                response = self.llm_handler.chat(messages)[0].outputs[0].text.strip()
            elif self.llm_handler.model_type in ["openai", "anthropic"]:
                response = self.llm_handler.chat(messages)
            try:
                json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
                validated = EUSData(**json_response)
            except json.JSONDecodeError or ValidationError as e:
                print(f"Error processing file {filename}: {e}")
                continue #todo better handling
            
            # Parse EUS response
            outputs.append({
                "id": filename,
                "model": self.llm_handler.model_type,
                **validated.dict()
            })

        # Save to csv and postgres
        self.save_outputs(outputs)
    
    def save_outputs(self, outputs):
        eus_df = pd.DataFrame(outputs)
        # Save CSV (append)
        self.save_dataframe(eus_df, self.output_fp, index=False)
        # Upsert to Postgres if requested
        self.upsert_dataframe(eus_df, "eus_procedures")
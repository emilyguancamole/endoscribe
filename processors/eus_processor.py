import json
from .base_processor import BaseProcessor
import pandas as pd

class EUSProcessor(BaseProcessor):
    def process_transcripts(self, filenames_to_process, transcripts_df):
        outputs = []
        # Prompt files for EUS
        prompt_field_definitions_fp = './prompts/eus/eus.txt'
        fewshot_examples_dir = "./prompts/eus/fewshot"

        for _, row in transcripts_df.iterrows():
            if filenames_to_process[0] != "all" and row["file"] not in filenames_to_process:
                continue

            cur_transcript = row["pred_transcript"]
            filename = row["file"]

            messages = self.build_messages(
                cur_transcript,
                system_prompt_fp=self.system_prompt_fp,
                prompt_field_definitions_fp=prompt_field_definitions_fp,
                fewshot_examples_dir=fewshot_examples_dir,
                prefix="eus"
            )

            response = self.llm_handler.chat(messages)[0].outputs[0].text.strip()
            try:
                json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
            except json.JSONDecodeError:
                continue
            
            # Parse EUS response
            outputs.append({
                "id": filename,
                "attending": "Llama4",  # placeholder
                "indications": json_response.get("indications", ""),
                "eus_findings": json_response.get("eus_findings", ""),
                "egd_findings": json_response.get("egd_findings", ""),
                "impressions": json_response.get("impressions", "")
            })

        # Save to csv and postgres
        self.save_outputs(outputs)
    
    def save_outputs(self, outputs):
        eus_df = pd.DataFrame(outputs)
        eus_df.to_csv(self.output_fp, index=False)
        
        if self.to_postgres:
            from db.postgres_writer import create_tables_if_not_exist, upsert_extracted_outputs
            create_tables_if_not_exist()
            
            # eus_df = self.convert_data_types(eus_df) # Currently, eus has only text data; add if number/typed data is added later
            if not eus_df.empty:
                upsert_extracted_outputs(eus_df, "eus_procedures")
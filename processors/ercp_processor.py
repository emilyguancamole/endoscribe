import json
from .base_processor import BaseProcessor
import pandas as pd
from data_models.data_models import ERCPData


class ERCPProcessor(BaseProcessor):
    def process_transcripts(self, filenames_to_process, transcripts_df):
        outputs = []
        # Prompt files for ERCP
        prompt_field_definitions_fp = './prompts/ercp/ercp.txt'
        fewshot_examples_dir = "./prompts/ercp/fewshot"

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
                prefix="ercp"
            )
            response = self.llm_handler.chat(messages)[0].outputs[0].text.strip()
            try:
                json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
                validated = ERCPData(**json_response)
            except json.JSONDecodeError:
                continue
            
            # Parse ERCP response - not in a sep file (as in col_processor)
            outputs.append({
                "id": filename,
                "attending": "Llama4", # placeholder
                **validated.dict()
            })

        # Save outputs to csv
        self.save_outputs(outputs)

    def save_outputs(self, outputs):
        ercp_df = pd.DataFrame(outputs)
        ercp_df.to_csv(self.output_fp, index=False)
        
        if self.to_postgres:
            from db.postgres_writer import create_tables_if_not_exist, upsert_extracted_outputs
            create_tables_if_not_exist()
            
            # ercp_df = self.convert_data_types(ercp_df) # Currently, ERCP has only text data; add if number/typed data is added later
            if not ercp_df.empty:
                upsert_extracted_outputs(ercp_df, "ercp_procedures")
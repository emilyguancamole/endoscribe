import json
from .base_processor import BaseProcessor
import pandas as pd
from data_models.data_models import EGDData


class EGDProcessor(BaseProcessor):
    def process_transcripts(self, filenames_to_process, transcripts_df):
        outputs = []
        # Prompt files
        prompt_field_definitions_fp = './prompts/egd/egd.txt'
        fewshot_examples_dir = "./prompts/egd/fewshot"

        for _, row in transcripts_df.iterrows():
            if filenames_to_process[0] != "all" and row["participant_id"] not in filenames_to_process:
                continue

            cur_transcript = row["pred_transcript"]
            filename = row["participant_id"]

            messages = self.build_messages(
                cur_transcript,
                system_prompt_fp=self.system_prompt_fp,
                prompt_field_definitions_fp=prompt_field_definitions_fp,
                fewshot_examples_dir=fewshot_examples_dir,
                prefix="egd"
            )
            print(f"Processing file: {filename} with {len(messages)} messages")
            print("")
            if self.llm_handler.model_type == "local": #! diff response processing for local vs openai
                response = self.llm_handler.chat(messages)[0].outputs[0].text.strip()
            elif self.llm_handler.model_type == "openai":
                response = self.llm_handler.chat(messages)
            try:
                json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
                validated = EGDData(**json_response)
            except json.JSONDecodeError:
                continue

            # Parse response - not in a sep file (as in col_processor)
            outputs.append({
                "id": filename,
                "model": self.llm_handler.model_type,
                **validated.dict()
            })

        # Save outputs to csv
        self.save_outputs(outputs)

    def save_outputs(self, outputs):
        egd_df = pd.DataFrame(outputs)
        egd_df.to_csv(self.output_fp, index=False)
        
        if self.to_postgres:
            from db.postgres_writer import create_tables_if_not_exist, upsert_extracted_outputs
            create_tables_if_not_exist()
            
            # ercp_df = self.convert_data_types(ercp_df) # Currently, EGD has only text data; add if number/typed data is added later
            if not egd_df.empty:
                upsert_extracted_outputs(egd_df, "egd_procedures")
import json
from .base_processor import BaseProcessor
import pandas as pd
from data_models.data_models import ERCPData, PEPRiskData


class ERCPProcessor(BaseProcessor):
    def extract_pep_from_transcript(self, transcript: str, filename: str = "live") -> dict:
        """
        Run PEP risk extraction on a single transcript and return a validated dict.
        This is a lightweight wrapper for server-side, one-off processing.
        # todo perhaps place this within pep_risk folder for modularity

        Returns a dict with keys: id, model, and the ERCPData fields.
        """
        prompt_field_definitions_fp = 'pep_risk/prompts/field_definitions.txt'
        fewshot_examples_dir = "pep_risk/prompts/fewshot"

        messages = self.build_messages(
            transcript,
            system_prompt_fp=self.system_prompt_fp,
            prompt_field_definitions_fp=prompt_field_definitions_fp,
            fewshot_examples_dir=fewshot_examples_dir,
            prefix="pep",
        )
        print("PEP messages:\n", messages)

        if self.llm_handler.model_type == "local":
            response = self.llm_handler.chat(messages)[0].outputs[0].text.strip()
        elif self.llm_handler.model_type == "openai":
            response = self.llm_handler.chat(messages)

        json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
        print("PEP extraction raw json result:\n", json_response)
        validated = PEPRiskData(**json_response)
        print("PEP extraction validated result:\n", validated)

        return {
            "id": filename,
            "model": self.llm_handler.model_type,
            **validated.dict(),
        }
    
    def process_transcripts(self, filenames_to_process, transcripts_df):
        outputs = []
        # Prompt files for ERCP
        prompt_field_definitions_fp = './prompts/ercp/ercp.txt'
        fewshot_examples_dir = "./prompts/ercp/fewshot"

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
                prefix="ercp"
            )
            if self.llm_handler.model_type == "local": #! diff response processing for local vs openai
                response = self.llm_handler.chat(messages)[0].outputs[0].text.strip()
            elif self.llm_handler.model_type == "openai":
                response = self.llm_handler.chat(messages)
            try:
                json_response = json.loads(response[response.find("{"): response.rfind("}") + 1])
                validated = ERCPData(**json_response)
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
        ercp_df.to_csv(self.output_fp, index=False)
        
        if self.to_postgres:
            from db.postgres_writer import create_tables_if_not_exist, upsert_extracted_outputs
            create_tables_if_not_exist()
            
            # ercp_df = self.convert_data_types(ercp_df) # Currently, ERCP has only text data; add if number/typed data is added later
            if not ercp_df.empty:
                upsert_extracted_outputs(ercp_df, "ercp_procedures")
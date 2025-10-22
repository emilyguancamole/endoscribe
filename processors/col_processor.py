import json
from .base_processor import BaseProcessor
from data_models.data_models import ColonoscopyData, PolypData
from typing import List, Dict
import pandas as pd

class ColProcessor(BaseProcessor):
    def build_polyp_messages(self, transcript: str, findings: str, polyp_count: str, system_prompt_fp: str) -> List[Dict[str, str]]:
        ''' Special function for polyp extraction, since it requires findings from colonoscopy 
        Currently hardcoding the files for system prompt and fewshot examples
        '''
        system_prompt = open(system_prompt_fp).read().replace(
            '{{prompt_field_definitions}}',
            open('./prompts/col/polyps.txt').read() #05_col_experiments/prompts/polyps.txt
        )
        messages = [{"role": "system", "content": system_prompt}]
    
        fewshot_examples = super().load_fewshot_examples("./prompts/col/fewshot", prefix="polyp")
        for ex in fewshot_examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        # Final, actual task
        messages.append({"role": "user", 
                        "content": f"""Extract details from the following transcript and findings. There are {polyp_count} polyps, so there should be {polyp_count} JSON objects in the array.\n\nTranscript:\n{transcript}\n\nFindings:\n{findings}"""
                        })

        return messages


    def process_transcripts(self, filenames_to_process, transcripts_df, max_attempts=2):
        col_outputs = []
        polyp_outputs = []

        for _, row in transcripts_df.iterrows():
            if filenames_to_process[0] != "all" and row["participant_id"] not in filenames_to_process:
                continue

            cur_transcript = row["pred_transcript"]
            filename = row["participant_id"]

            # Colonoscopy-level processing with retry
            col_json = None
            for attempt in range(max_attempts):
                col_messages = self.build_messages(
                    cur_transcript,
                    system_prompt_fp=self.system_prompt_fp,
                    prompt_field_definitions_fp='./prompts/col/colonoscopies.txt',
                    fewshot_examples_dir="./prompts/col/fewshot",
                    prefix="col"
                )
                if self.llm_handler.model_type == "local":
                    col_response = self.llm_handler.chat(col_messages)[0].outputs[0].text.strip()
                elif self.llm_handler.model_type in ["openai", "anthropic"]:
                    col_response = self.llm_handler.chat(col_messages)
                try:
                    col_json = json.loads(col_response[col_response.find("{"): col_response.rfind("}") + 1])
                    col_outputs.append(self.parse_validate_colonoscopy_response(col_json, filename))
                    break
                except (json.JSONDecodeError, ValueError):
                    if attempt == 1:
                        col_json = None
            if not col_json:
                continue

            # Polyp-level processing with retry
            polyps_json = None
            for attempt in range(max_attempts):
                polyp_messages = self.build_polyp_messages(
                    cur_transcript,
                    findings=col_json.get("findings", ""),
                    polyp_count=col_json.get("polyp_count", 0),
                    system_prompt_fp=self.system_prompt_fp
                )
                if self.llm_handler.model_type == "local":
                    polyp_response = self.llm_handler.chat(polyp_messages)[0].outputs[0].text.strip()
                elif self.llm_handler.model_type in ["openai", "anthropic"]:
                    polyp_response = self.llm_handler.chat(polyp_messages)
                try:
                    polyps_json = json.loads(polyp_response)
                    polyp_outputs.extend(self.parse_validate_polyp_response(polyps_json, filename))
                    break
                except (json.JSONDecodeError, ValueError):
                    if attempt == 1:
                        polyps_json = None

        # Save outputs to csv files
        self.save_outputs(col_outputs, polyp_outputs)

    def parse_validate_colonoscopy_response(self, col_json, filename, data_model=ColonoscopyData):
        """ Use data_model to validate JSON fields + types, then parse + return as dict """
        try:
            col_data = data_model(**col_json)
        except Exception as e:
            raise ValueError(f"Validation failed for colonoscopy data for {filename}: {e}")

        col_json = col_data.dict()
        return {
            "id": filename,
            "model": self.llm_handler.model_type,
            **col_json
        }
        

    def parse_validate_polyp_response(self, polyps_json: List, filename: str, data_model=PolypData):
        """ Use data_model to validate JSON fields + types, then parse + return as list of dicts """
        try:
            for polyp in polyps_json:
                data_model(**polyp)  # This will raise an error if validation fails
        except Exception as e:
            raise ValueError(f"Validation failed for polyp data for {filename}: {e}")
        
        return [
            {
                "col_id": filename,
                **polyp
            } for polyp in polyps_json
        ]
        
    
    def convert_data_types(self, col_df, polyp_df):
        """Convert data types for PostgreSQL insertion"""
        # Convert colonoscopy data types
        col_df = col_df.copy()
        col_df['bbps_right'] = pd.to_numeric(col_df['bbps_right'], errors='coerce')
        col_df['bbps_transverse'] = pd.to_numeric(col_df['bbps_transverse'], errors='coerce')
        col_df['bbps_left'] = pd.to_numeric(col_df['bbps_left'], errors='coerce')
        col_df['bbps_total'] = pd.to_numeric(col_df['bbps_total'], errors='coerce')
        col_df['polyp_count'] = pd.to_numeric(col_df['polyp_count'], errors='coerce')
        
        # Convert impressions list to string if needed
        if 'impressions' in col_df.columns:
            col_df['impressions'] = col_df['impressions'].apply(
                lambda x: str(x) if isinstance(x, list) else x
            )
        
        # Convert polyp data types
        polyp_df = polyp_df.copy()
        polyp_df['size_min_mm'] = pd.to_numeric(polyp_df['size_min_mm'], errors='coerce')
        polyp_df['size_max_mm'] = pd.to_numeric(polyp_df['size_max_mm'], errors='coerce')
        polyp_df['nice_class'] = pd.to_numeric(polyp_df['nice_class'], errors='coerce')
        
        #? Convert boolean fields
        polyp_df['resection_performed'] = polyp_df['resection_performed'].apply(
            lambda x: True if str(x).lower() in ['true', 'yes'] else False if str(x).lower() in ['false', 'no'] else None
        )
        
        return col_df, polyp_df

    def save_outputs(self, col_outputs, polyp_outputs):
        col_df = pd.DataFrame(col_outputs)
        polyp_df = pd.DataFrame(polyp_outputs)
        # Save to csv
        col_df.to_csv(self.output_fp.replace(".csv", "_colonoscopies.csv"), index=False)
        polyp_df.to_csv(self.output_fp.replace(".csv", "_polyps.csv"), index=False)

        # Save to postgres if needed
        if self.to_postgres:
            from db.postgres_writer import create_tables_if_not_exist, upsert_extracted_outputs
            create_tables_if_not_exist()
            
            # Convert data types before inserting
            col_df, polyp_df = self.convert_data_types(col_df, polyp_df)
            
            if not col_df.empty:
                upsert_extracted_outputs(col_df, "colonoscopy_procedures")
            if not polyp_df.empty:
                upsert_extracted_outputs(polyp_df, "polyps")
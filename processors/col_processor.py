import json
from .base_processor import BaseProcessor
from typing import List, Dict
import pandas as pd

class ColProcessor(BaseProcessor):
    def build_polyp_messages(self, transcript: str, findings: str, polyp_count: str, system_prompt_fp: str) -> List[Dict[str, str]]:
        ''' Special function for polyp extraction, since it requires findings from colonoscopy 
        Currently hardcoding the files for system prompt and fewshot examples'''
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


    def process_transcripts(self, filenames_to_process, transcripts_df):
        col_outputs = []
        polyp_outputs = []

        for _, row in transcripts_df.iterrows():
            if filenames_to_process[0] != "all" and row["file"] not in filenames_to_process:
                continue

            cur_transcript = row["pred_transcript"]
            filename = row["file"]

            # Colonoscopy-level processing
            col_messages = self.build_messages( 
                cur_transcript,
                system_prompt_fp=self.system_prompt_fp,
                prompt_field_definitions_fp='./prompts/col/colonoscopies.txt',
                fewshot_examples_dir="./prompts/col/fewshot",
                prefix="col"
            )
            col_response = self.llm_handler.chat(col_messages)[0].outputs[0].text.strip()
            try:
                col_json = json.loads(col_response[col_response.find("{"): col_response.rfind("}") + 1])
                col_outputs.append(self.parse_colonoscopy_response(col_json, filename))
            except json.JSONDecodeError:
                continue

            # Polyp-level processing
            polyp_messages = self.build_polyp_messages(
                cur_transcript,
                findings=col_json.get("findings", ""),
                polyp_count=col_json.get("polyp_count", 0),
                system_prompt_fp=self.system_prompt_fp
            )
            print("Polyp messages:\n", polyp_messages)
            
            polyp_response = self.llm_handler.chat(polyp_messages)[0].outputs[0].text.strip()
            try:
                polyps_json = json.loads(polyp_response)
                # todo maybe add?? parse into int/float here... but prob better in a loading data script before postgres
                polyp_outputs.extend(self.parse_polyp_response(polyps_json, filename))
            except json.JSONDecodeError:
                continue

        # Save outputs to csv files
        self.save_outputs(col_outputs, polyp_outputs)

    def parse_colonoscopy_response(self, col_json, filename):
        return {
            "id": filename,
            "attending": "Llama 4", #! placeholder
            "indications": col_json.get("indications", ""),
            "last_colonoscopy": col_json.get("last_colonoscopy", ""),
            "bbps_simple": col_json.get("bbps_simple", ""),
            "bbps_right": col_json.get("bbps_right", ""),
            "bbps_transverse": col_json.get("bbps_transverse", ""),
            "bbps_left": col_json.get("bbps_left", ""),
            "bbps_total": col_json.get("bbps_total", ""),
            "extent": col_json.get("extent", ""),
            "findings": col_json.get("findings", ""),
            "polyp_count": col_json.get("polyp_count", ""),
            "impressions": col_json.get("impressions", []),
        }

    def parse_polyp_response(self, polyps_json, filename):
        return [
            {
                "col_id": filename,
                "size_min_mm": polyp.get("size_min_mm", ""),
                "size_max_mm": polyp.get("size_max_mm", ""),
                "location": polyp.get("location", ""),
                "resection_performed": polyp.get("resection_performed", ""),
                "resection_method": polyp.get("resection_method", ""),
                "nice_class": polyp.get("nice_class", ""),
                "jnet_class": polyp.get("jnet_class", ""),
                "paris_class": polyp.get("paris_class", ""),
            }
            for polyp in polyps_json
        ]

    def save_outputs(self, col_outputs, polyp_outputs):
        col_df = pd.DataFrame(col_outputs)
        polyp_df = pd.DataFrame(polyp_outputs)
        col_df.to_csv(self.output_fp.replace(".csv", "_colonoscopies.csv"), index=False)
        polyp_df.to_csv(self.output_fp.replace(".csv", "_polyps.csv"), index=False)
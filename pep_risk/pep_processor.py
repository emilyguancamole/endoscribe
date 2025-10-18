import os
from typing import Dict, List
import pandas as pd

class PEPProcessor:
    def __init__(self, procedure_type, system_prompt_fp, output_fp, llm_handler, to_postgres):
        self.procedure_type = procedure_type
        self.system_prompt_fp = system_prompt_fp
        self.output_fp = output_fp
        self.llm_handler = llm_handler
        self.to_postgres = to_postgres

    def load_transcripts_to_df(self, filepath):
        transcripts_df = pd.read_csv(filepath)
        transcripts_df["file"] = transcripts_df["file"].astype(str)
        return transcripts_df
    
    def load_fewshot_examples(self, folder: str, prefix: str) -> list:
        """
        Load few-shot examples from a folder. Each example must have filenames as: <prefix>_1_user.txt and <prefix>_1_assistant.txt.

        folder (str): path to the folder containing the example files.
        prefix (str): indicates the type of example: "col" or "polyp".

        Returns: list of dictionaries with keys "user" and "assistant".
        """
        examples = []
        user_files = sorted([f for f in os.listdir(folder) if f.startswith(prefix) and "_user" in f])

        for user_file in user_files:
            base = user_file.replace("_user.txt", "")
            assistant_file = f"{base}_assistant.txt"
            with open(os.path.join(folder, user_file), "r") as u:
                user_content = u.read().strip()
            with open(os.path.join(folder, assistant_file), "r") as a:
                assistant_content = a.read().strip()
            examples.append({"user": user_content, "assistant": assistant_content})

        return examples

    def build_messages(self, transcript, system_prompt_fp: str, prompt_field_definitions_fp: str, fewshot_examples_dir: str, prefix) -> List[Dict[str, str]]:

        system_prompt = open(self.system_prompt_fp).read().replace(
            '{{prompt_field_definitions}}',
            open(prompt_field_definitions_fp).read()
        )
        messages = [{"role": "system", "content": system_prompt}]

        fewshot_examples = self.load_fewshot_examples(fewshot_examples_dir, prefix)
        for ex in fewshot_examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        # Final, actual task
        messages.append({"role": "user", 
                    "content": f"""Extract procedure entities from the following transcript:\n\n{transcript}"""
                    })
        return messages


    def process_transcripts(self, filenames_to_process, transcripts_df):
        pass
from abc import ABC, abstractmethod
import os
from typing import Dict, List, Optional
import pandas as pd

class BaseProcessor:
    def __init__(self, procedure_type, system_prompt_fp, output_fp, llm_handler, to_postgres=False):
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

    def save_dataframe(self, df: 'pd.DataFrame', output_fp: str, index: bool = False) -> None:
        """
        Generic CSV append helper: ensures output directory exists and appends the dataframe
        to `output_fp`. If the file doesn't exist, writes header.
        """
        out_dir = os.path.dirname(output_fp)
        os.makedirs(out_dir, exist_ok=True)
        write_header = not os.path.exists(output_fp)
        df.to_csv(output_fp, index=index, mode='a', header=write_header)

    def upsert_dataframe(self, df: 'pd.DataFrame', table_name: str) -> None:
        """
        Upsert dataframe to Postgres if `to_postgres` is enabled for this processor.
        Processors should perform any necessary type conversions before calling this.
        """
        if not self.to_postgres:
            return
        if df.empty:
            return
        from db.postgres_writer import create_tables_if_not_exist, upsert_extracted_outputs
        create_tables_if_not_exist()
        upsert_extracted_outputs(df, table_name)

    def build_messages(self, transcript, prompt_field_definitions_fp: str, fewshot_examples_dir: Optional[str]=None, prefix: Optional[str]=None) -> List[Dict[str, str]]:
        # Verify files exist before opening
        if not os.path.exists(self.system_prompt_fp):
            raise FileNotFoundError(f"System prompt file not found: {self.system_prompt_fp} (cwd: {os.getcwd()})")
        if not os.path.exists(prompt_field_definitions_fp):
            raise FileNotFoundError(f"Prompt field definitions file not found: {prompt_field_definitions_fp} (cwd: {os.getcwd()})")
        
        system_prompt = open(self.system_prompt_fp).read().replace(
            '{{prompt_field_definitions}}',
            open(prompt_field_definitions_fp).read()
        )
        print(f"\nLoaded system prompt from: {self.system_prompt_fp} with field definitions from: {prompt_field_definitions_fp}")
        messages = [{"role": "system", "content": system_prompt}]

        # Load and add few-shot examples
        if fewshot_examples_dir and os.path.exists(fewshot_examples_dir): #! TODO add few shot
            print(f"Loading few-shot examples from: {fewshot_examples_dir} with prefix: {prefix}")
            # fewshot_examples = self.load_fewshot_examples(fewshot_examples_dir, prefix)
            # for ex in fewshot_examples:
            #     messages.append({"role": "user", "content": ex["user"]})
            #     messages.append({"role": "assistant", "content": ex["assistant"]})
        
        messages.append({
            "role": "user", 
            "content": f"""Extract procedure entities from the following transcript:\n\n{transcript}"""
        })
        print("###############Built messages for LLM for processor", self.procedure_type)
        for msg in messages:
            print(f"\n###Message: {msg['content'][:300]}...")
        return messages


    @abstractmethod
    def process_transcripts(self, filenames_to_process, transcripts_df):
        pass
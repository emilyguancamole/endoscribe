import os
from typing import List, Dict

def load_fewshot_examples(fp: str) -> str:
    with open(fp, 'r') as f:
        return f.read()

def build_polyp_messages(fewshot: str, transcript: str, findings: str):
    return [
        {"role": "system", "content": "You are an expert GI assistant..."},
        {"role": "user", "content": fewshot},
        {"role": "user", "content": f"Transcript: {transcript}\nFindings: {findings}"},
    ]

def load_fewshot_examples(folder: str, prefix: str) -> list:
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


def build_polyp_messages(transcript: str, findings: str, polyp_count: str, system_prompt_fp: str) -> List[Dict[str, str]]:
    ''' Special function for polyp extraction, since it requires findings from colonoscopy '''
    system_prompt = open(system_prompt_fp).read().replace(
        '{{prompt_field_definitions}}',
        open('05_col_experiments/prompts/polyps.txt').read()
    )
    messages = [{"role": "system", "content": system_prompt}]

    fewshot_examples = load_fewshot_examples("05_col_experiments/prompts/fewshot", prefix="polyp")
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    # Final, actual task
    messages.append({"role": "user", 
                    "content": f"""Extract details from the following transcript and findings. There are {polyp_count} polyps, so there should be {polyp_count} JSON objects in the array.\n\nTranscript:\n{transcript}\n\nFindings:\n{findings}"""
                    })

    return messages


def build_llm_messages(transcript, system_prompt_fp: str, prompt_field_definitions_fp: str, fewshot_examples_dir: str, prefix) -> List[Dict[str, str]]:
    """Build messages for the LLM based on the transcript and system prompt.
    Args:
        transcript (str): The medical transcript to process.
        system_prompt_fp (str): Path to the system prompt file.
        prompt_field_definitions_fp (str): Path to the prompt field definitions file.
        fewshot_examples_dir (str): Directory containing few-shot examples.
        prefix (str): Prefix to find files for few-shot examples: "ercp", "eus", "col", "polyp".
    """
    system_prompt = open(system_prompt_fp).read().replace(
        '{{prompt_field_definitions}}',
        open(prompt_field_definitions_fp).read()
    )
    messages = [{"role": "system", "content": system_prompt}]

    fewshot_examples = load_fewshot_examples(f"{fewshot_examples_dir}", prefix=prefix)
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    # Final, actual task
    messages.append({"role": "user", 
                    "content": f"""Extract procedure entities from the following transcript:\n\n{transcript}"""
                    })
    return messages
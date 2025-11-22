import json
import os
import re
from typing import Optional, Dict, Any

from llm.llm_client import LLMClient


REVIEWER_SCHEMA_TOP_KEYS = {'updated_fields', 'updated_sections', 'accept_rendered_note', 'final_note', 'deltas', 'confidence', 'warnings'}


def _extract_json(text: str) -> Optional[str]:
    """Try to find a JSON object in the text output from the model.
    Returns the JSON substring or None.
    """
    if not text:
        return None
    # Try to find the first {...} or [ ... ] block that parses as JSON
    # Search for the first balanced braces block
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    # fallback: try to locate a JSON-looking substring between first '[' and last ']'
    sq = text.find('[')
    if sq != -1:
        try:
            candidate = text[text.find('['):text.rfind(']')+1]
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    return None


def _basic_validate_schema(parsed: Dict[str, Any]) -> bool:
    # ensure top-level keys are present (some may be optional)
    if not isinstance(parsed, dict):
        return False
    # At minimum expect accept_rendered_note and final_note or updated_sections
    if 'accept_rendered_note' not in parsed and 'final_note' not in parsed and 'updated_sections' not in parsed:
        return False
    # keys should be subset of allowed
    if not set(parsed.keys()).issubset(REVIEWER_SCHEMA_TOP_KEYS):
        # allow additional keys but warn
        pass
    return True


def run_reviewer_on_package(package_fp: str, llm_handler: Optional[LLMClient] = None, dry_run: bool = False) -> Dict[str, Any]:
    """Read package JSON, call reviewer LLM (or mock), parse reviewer JSON, write output file.

    Returns dict with parsed reviewer output and path to reviewer JSON file.
    """
    if not os.path.exists(package_fp):
        raise FileNotFoundError(package_fp)

    with open(package_fp, 'r') as f:
        package = json.load(f)

    sample_id = package.get('sample_id') or os.path.splitext(os.path.basename(package_fp))[0]

    # Build system + user messages
    system_msg = (
        "You are a concise medical report reviewer. You will receive the original transcript, the extracted fields, "
        "and the rendered report sections. Your job is to (1) check for consistency between the transcript and the report, "
        "(2) correct grammar and readability issues in the rendered sections, and (3) propose structured field updates when the transcript supports them. "
        "Return ONLY a single JSON object (no surrounding text) matching the schema: {\n"
        "  \"updated_fields\": { ... },\n"
        "  \"updated_sections\": { ... },\n"
        "  \"accept_rendered_note\": true|false,\n"
        "  \"final_note\": \"...\",\n"
        "  \"deltas\": [ {\"type\": \"field|section\", \"name\": \"...\", \"old\": \"...\", \"new\": \"...\", \"justification\": \"quote from transcript\", \"transcript_evidence\": [ {\"text\": \"...\"} ] } ],\n"
        "  \"confidence\": 0.0-1.0,\n"
        "  \"warnings\": [ ... ]\n"
        "}\n"
    )

    user_msg_parts = [
        "Package JSON:\n",
        json.dumps({
            'sample_id': package.get('sample_id'),
            'meta': package.get('meta'),
            'original_transcript': package.get('original_transcript'),
            'extracted_fields_normalized': package.get('extracted_fields', {}).get('normalized'),
            'extracted_fields_raw': package.get('extracted_fields', {}).get('raw'),
            'rendered_sections': package.get('rendered_sections')
        }, indent=2)
    ]

    user_msg = "\n".join([p if isinstance(p, str) else json.dumps(p) for p in user_msg_parts])

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    response_text = None
    if dry_run or llm_handler is None:
        # Produce a conservative mock reviewer that accepts the rendered note and returns no field updates
        mock = {
            'updated_fields': {},
            'updated_sections': {},
            'accept_rendered_note': True,
            'final_note': package.get('rendered_note'),
            'deltas': [],
            'confidence': 0.9,
            'warnings': []
        }
        response_text = json.dumps(mock)
    else:
        print("Reviewer messages:\n", messages)
        # Send to LLM
        try:
            if llm_handler.model_type == "local":
                response_text = llm_handler.chat(messages)[0].outputs[0].text.strip()
            elif llm_handler.model_type in ["openai", "anthropic"]:
                response_text = llm_handler.chat(messages)
      # response_text = llm_handler.chat(messages)

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    extracted = _extract_json(response_text)
    if not extracted:
        # write raw response for inspection
        out_fp = package_fp.replace('.json', '_reviewer_raw.txt')
        with open(out_fp, 'w') as rf:
            rf.write(response_text)
        raise ValueError(f"Could not extract JSON from reviewer output; raw output saved to {out_fp}")

    parsed = json.loads(extracted)

    # Basic validation
    if not _basic_validate_schema(parsed):
        out_fp = package_fp.replace('.json', '_reviewer_raw.txt')
        with open(out_fp, 'w') as rf:
            rf.write(response_text)
        raise ValueError(f"Reviewer JSON failed basic schema validation; raw output saved to {out_fp}")

    reviewer_fp = package_fp.replace('.json', '_reviewer.json')
    with open(reviewer_fp, 'w') as wf:
        json.dump(parsed, wf, indent=2, ensure_ascii=False)

    return {'reviewer_fp': reviewer_fp, 'parsed': parsed}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('package_fp')
    parser.add_argument('--model_config', choices=['local_llama', 'openai_gpt4o', 'anthropic_claude'],
                       default='local_llama', help="Predefined model configuration to use")
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    # Initialize LLM
    llm_handler = None
    if not args.dry_run:
        llm_handler = LLMClient.from_config(args.model_config)
    res = run_reviewer_on_package(args.package_fp, llm_handler=llm_handler, dry_run=args.dry_run)
    print('Reviewer output written to', res['reviewer_fp'])

import json
import os
import re
from typing import Optional, Dict, Any

from llm.llm_client import LLMClient


REVIEWER_SCHEMA_TOP_KEYS = {'updated_fields', 'updated_sections', 'accept_rendered_note', 'final_note', 'deltas', 'confidence', 'warnings'}


def _extract_json(text: str) -> Optional[str]:
    """Find the first valid JSON object/array in `text` and return it as a
    substring. This uses json.JSONDecoder.raw_decode to robustly locate a
    decodable JSON value. Returns None if no valid JSON is found.
    """
    if not text:
        return None

    decoder = json.JSONDecoder()

    # Try scanning forward from the first brace or bracket characters
    start_chars = ['{', '[']
    for idx, ch in enumerate(text):
        if ch not in start_chars:
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
            # raw_decode returns the python object and the index where parsing ended
            return text[idx: idx + end]
        except json.JSONDecodeError:
            # not a valid JSON starting here; continue scanning
            continue
    print("WARNING: no valid JSON found in reviewer response")
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


def _normalize_llm_response(resp: Any) -> str:
    """Coerce various llm_handler.chat return types into a single text string.
    Handles common dict/list shapes from OpenAI/Anthropic wrappers.
    """
    # simple string
    if isinstance(resp, str):
        return resp.strip()

    # openai-like dict with 'choices'
    if isinstance(resp, dict):
        print("ISINSTANCE DICT")
        if 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
            c0 = resp['choices'][0]
            if isinstance(c0, dict):
                # chat completion style
                if 'message' in c0 and isinstance(c0['message'], dict):
                    content = c0['message'].get('content') or c0['message'].get('text')
                    if isinstance(content, str):
                        return content.strip()
                # text-style
                if 'text' in c0 and isinstance(c0['text'], str):
                    return c0['text'].strip()
        # other common keys
        for key in ('output', 'outputs', 'text', 'content', 'response'):
            if key in resp:
                val = resp[key]
                if isinstance(val, str):
                    return val.strip()
                if isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, str):
                        return first.strip()
                    if isinstance(first, dict):
                        for k in ('text', 'content'):
                            if k in first and isinstance(first[k], str):
                                return first[k].strip()
        # fallback: serialize the dict
        try:
            return json.dumps(resp, ensure_ascii=False)
        except Exception:
            return str(resp)

    # list-like
    if isinstance(resp, list):
        print("ISINSTANCE LIST")
        if not resp:
            return ''
        first = resp[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ('text', 'content', 'output'):
                if key in first and isinstance(first[key], str):
                    return first[key].strip()
            if 'outputs' in first and isinstance(first['outputs'], list) and first['outputs']:
                out0 = first['outputs'][0]
                if isinstance(out0, dict) and 'text' in out0 and isinstance(out0['text'], str):
                    return out0['text'].strip()
        # fallback join
        pieces = []
        for item in resp:
            if isinstance(item, str):
                pieces.append(item)
            else:
                try:
                    pieces.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    pieces.append(str(item))
        return '\n'.join(pieces)

def run_reviewer_on_package(package_fp: str, llm_handler: Optional[LLMClient] = None, dry_run: bool = False, debug: bool = False) -> Dict[str, Any]:
    """Read package JSON, call reviewer LLM (or mock), parse reviewer JSON, write output file.

    Returns dict with parsed reviewer output and path to reviewer JSON file.
    """
    if not os.path.exists(package_fp):
        raise FileNotFoundError(package_fp)

    with open(package_fp, 'r') as f:
        package = json.load(f)

    sample_id = package.get('sample_id') or os.path.splitext(os.path.basename(package_fp))[0]

    # Build system + user messages
    system_msg = open(os.path.join(os.path.dirname(__file__), '..', 'prompts', 'ercp', 'reviewer_prompt.txt')).read()

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
    if llm_handler is not None and getattr(llm_handler, 'model_type', None) not in ['openai', 'anthropic']:
        raise ValueError("Reviewer only supports 'openai', 'anthropic'")

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
        try:
            raw_resp = llm_handler.chat(messages)
            response_text = _normalize_llm_response(raw_resp)

        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    extracted = _extract_json(response_text)
    if not extracted: # something went wrong, write raw response for inspection
        out_fp = package_fp.replace('.json', '_reviewer_raw.txt')
        with open(out_fp, 'w') as rf:
            rf.write("--- MESSAGES SENT ---\n")
            rf.write(json.dumps(messages, indent=2, ensure_ascii=False))
            rf.write("\n--- RAW RESPONSE ---\n")
            rf.write(str(response_text))
        raise ValueError(f"Could not extract JSON; raw output saved to {out_fp}")
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
    parser.add_argument('--model_config', choices=['openai_gpt4o', 'anthropic_claude'],
                       default='openai_gpt4o', help="Predefined model configuration to use")
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Save debug request/response on failures')
    args = parser.parse_args()

    # Initialize LLM
    llm_handler = None
    if not args.dry_run:
        llm_handler = LLMClient.from_config(args.model_config)
    res = run_reviewer_on_package(args.package_fp, llm_handler=llm_handler, dry_run=args.dry_run, debug=args.debug)
    print('Reviewer output written to', res['reviewer_fp'])

import argparse
import pyjson5 as json5
import os
import sys
from pathlib import Path

_here = Path(__file__).parent
_root = _here.parent.parent
sys.path.insert(0, str(_root))

from templating.generate_from_fields import (
    load_fields_config,
    build_report_sections,
    merge_configs,
    generate_prompt_text,
    generate_base_yaml,
    generate_pydantic_model,
)
import pandas as pd
from central.drafters.ercp import ERCPDrafter


def _coerce_bool_strings(obj):
    """For demo data in json files, convert boolean-like strings to actual booleans in-place.
    'true'/'false' (case-insensitive) and 'yes'/'no'.
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _coerce_bool_strings(v)
        return obj
    if isinstance(obj, list):
        return [_coerce_bool_strings(v) for v in obj]
    if isinstance(obj, str):
        lv = obj.strip().lower()
        if lv == 'true' or lv == 'false':
            return lv == 'true'
        if lv == 'yes' or lv == 'no':
            return lv == 'yes'
        if lv in ('none', 'unknown'):
            return None
    return obj

def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc', default='ercp_base', help='Procedure type or path to yaml')
    parser.add_argument('--modules', default=None, help='Comma-separated module ids to merge into the base, e.g. "0.2,0.4"')
    parser.add_argument('--demo_data_json', default='ercp_base.json5', help='Demo data json file name under demo_data/')
    args = parser.parse_args()
    print("Procedure type:", args.proc)
    # Search for matching fields YAML under prompts/ercp/yaml
    yaml_root = _root / 'templating' / 'prompts' / 'ercp' / 'yaml'
    fields_path = None
    config = None

    # If direct path, use it
    if os.path.exists(args.proc):
        fields_path = Path(args.proc)
        config = load_fields_config(str(fields_path))
    else: # Match by meta.procedure_type or meta.module_id
        candidates = list(yaml_root.rglob('*.yaml')) if yaml_root.exists() else []
        for cand in candidates:
            cand_cfg = load_fields_config(str(cand))
            meta = cand_cfg.get('meta', {}) or {}
            if meta.get('procedure_type') == args.proc or meta.get('module_id') == args.proc:
                fields_path = cand
                config = cand_cfg
                break

    # Fallback to base fields if nothing matched
    if fields_path is None:
        fields_path = yaml_root / 'fields_base.yaml'
        config = load_fields_config(str(fields_path))

    # If modules specified, merge them into base
    if args.modules:
        module_ids = [m.strip() for m in args.modules.split(',') if m.strip()]
        # ensure base_config is the base
        base_path = yaml_root / 'fields_base.yaml'
        base_config = load_fields_config(str(base_path))
        for mid in module_ids:
            # find module file by id
            modules_dir = yaml_root / 'modules'
            mod_file = None
            if modules_dir.exists():
                matches = list(modules_dir.glob(f"{mid}_*.yaml"))
                if matches:
                    mod_file = matches[0]
            if not mod_file:
                candidate = modules_dir / f"{mid}.yaml"
                if candidate.exists():
                    mod_file = candidate
            if not mod_file:
                print(f"Warning: module {mid} not found; skipping")
                continue
            addon_cfg = load_fields_config(str(mod_file))
            base_config = merge_configs(base_config, addon_cfg)
        # replace config and fields_path with the merged result
        config = base_config
        fields_path = None

    drafter_path = _root / 'drafters' / 'procedures'
    procedure_meta = config.get('meta', {'procedure_group': 'ercp'})
    proc_group = procedure_meta.get('procedure_group', 'ercp')
    
    field_count = sum(len(group.get('fields', [])) for group in config.get('field_groups', {}).values())
    group_count = len(config.get('field_groups', {}))
    print(f"    Loaded {group_count} field groups with {field_count} total fields")
    
    # Generate prompt and base template
    proc_type = config.get('meta', {}).get('procedure_type', None)

    # write prompt
    prompt_text = generate_prompt_text(config)
    prompt_output = _here / f'demo_output/generated_{proc_type}_prompt.txt'
    with open(prompt_output, 'w') as f:
        f.write(prompt_text)

    # write base (drafter) yaml
    procedure_meta = config.get('meta', {'procedure_group': proc_group, 'procedure_type': proc_type})
    base_yaml = generate_base_yaml(config, procedure_meta)
    drafter_output = Path(drafter_path) / proc_group / f'generated_{proc_type}.yaml'
    os.makedirs(os.path.dirname(drafter_output), exist_ok=True)
    with open(drafter_output, 'w') as f:
        f.write(base_yaml)

    # Generate and write pydantic model
    model_code = generate_pydantic_model(config, f"{proc_type.replace('_', ' ').title().replace(' ', '')}Data")
    model_output = _here / f'demo_output/generated_{proc_type}_model.py'
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    with open(model_output, 'w') as f:
        f.write(model_code)
    print(f"    Generated prompt: {prompt_output}")
    print(f"    Generated base template: {drafter_output}")
    
    # Load demo data json
    with open(_here / "demo_data" / args.demo_data_json, "r") as f:
        demo_data = json5.load(f)
    demo_data = _coerce_bool_strings(demo_data)
    print(f"\nUsing demo data file ({args.demo_data_json})...")
    # Ensure all fields declared in fields config are present in test_data
    for group in config.get('field_groups', {}).values():
        for f in group.get('fields', []):
            fname = f.get('name')
            if fname and fname not in demo_data:
                # set booleans to False by default if type is boolean, else None
                ftype = f.get('type', 'string')
                if ftype == 'boolean':
                    demo_data[fname] = False
                elif ftype in ('integer', 'number', 'float'):
                    demo_data[fname] = -1
                else:
                    demo_data[fname] = None
    
    # Render using the generated base template and write to txt
    temp_yaml_path = drafter_output
    rendered = build_report_sections(str(temp_yaml_path), demo_data)

    # Instantiate an ERCP drafter for recommendations
    try:
        # df structure expected by Drafter
        sample_id = 'demo_sample'
        pred_df = pd.DataFrame([demo_data], index=[sample_id])
        drafter = ERCPDrafter(sample_id, pred_df)
        recommendations = drafter.construct_recommendations() or []
        if recommendations:
            rendered['recommendations'] = '\n'.join([f"{i}. {r}" for i, r in enumerate(recommendations, start=1)])
    except Exception:
        print("Error generating recommendations")
        pass

    output_path = _here/"demo_output/demo_ercp_report.txt"
    parts = []
    for section_name, content in rendered.items():
        parts.append(f"== {section_name.upper()} ==\n\n{content.strip()}\n")
    output_text = "\n".join(parts).strip() + "\n"
    with open(output_path, "w") as out_f:
        out_f.write(output_text)
    print(f"\nRendered test procedure note written to: {output_path}\n")

if __name__ == "__main__":
    ''' 
    To run demo: 
    python templating/demo/demo_ercp_yaml_pipeline.py
    '''
    demo()

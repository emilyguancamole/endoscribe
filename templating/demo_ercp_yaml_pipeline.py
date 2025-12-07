#!/usr/bin/env python3
"""
End-to-end demo of YAML-driven field-to-report pipeline.

This script demonstrates:
1. Generating prompts and templates from fields.yaml
2. Rendering a report using the generated templates
3. Comparing with manually created templates

Usage:
    python reporting/demo_yaml_pipeline.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
_here = Path(__file__).parent
_root = _here.parent
sys.path.insert(0, str(_root))

from templating.generate_from_fields import load_fields_config, generate_prompt_text, generate_base_yaml, generate_single
from templating.drafter_engine import build_report_sections
import yaml

def demo():
    print("=" * 80)
    print("YAML-DRIVEN FIELD-TO-REPORT PIPELINE DEMO")
    print("=" * 80)
    
    # Step 1: Load field definitions
    print("\n📄 Step 1: Loading field definitions from fields.yaml...")
    # Use the ERCP base fields currently in repo
    fields_path = _root / "prompts/ercp/yaml/fields_base.yaml"
    config = load_fields_config(str(fields_path))
    
    field_count = sum(len(group.get('fields', [])) for group in config.get('field_groups', {}).values())
    group_count = len(config.get('field_groups', {}))
    print(f"   ✓ Loaded {group_count} field groups with {field_count} total fields")
    
    # Step 2: Generate prompt and base template using the single-generator helper
    print("\nStep 2: Generating LLM extraction prompt...")
    proc_type = config.get('meta', {}).get('procedure_type', 'ercp_base')

    # generate_single will write the prompt and base files to specific locations.
    generate_single(str(fields_path), proc_type)

    # Recompute the output paths the generator uses so we can reference them.
    base_name = os.path.basename(str(fields_path)).replace('.yaml', '')
    prompt_output = Path(str(fields_path)).parent.joinpath(f'generated_{proc_type}_prompt.txt')
    procedure_meta = config.get('meta', {'procedure_group': 'ercp'})
    proc_group = procedure_meta.get('procedure_group', 'ercp')
    base_output = Path('prompts') / proc_group / f'generated_{proc_type}.yaml'
    print(f"   ✓ Generated prompt: {prompt_output}")
    print(f"   ✓ Generated base template: {base_output}")
    
    test_data = {
        "age": 58,
        "sex": "female",
        "chief_complaints": "recurrent bile duct stones",
        "symptoms_duration": "1 month",
        "symptoms_narrative": "Intermittent right upper quadrant pain for one month.",
        "negative_history": "jaundice",
        "past_medical_history": "Cholecystectomy 2 years ago.",
        "current_medications": "None",
        "family_history": "No relevant family history.",
        "social_history": "patient has smoked for ten years",
        "medications": "unknown",
        "monitoring": "Cardiac and pulse oximetry monitoring",
        "duodenoscope_type": "TJF-Q190V",
        "grade_of_ercp": 2,
        "pd_cannulation_status": "not_attempted",
        "pd_cannulation": "not_attempted",
        "cannulation_success": True,
        "lactated_ringers": True,
        "rectal_indomethacin": False,
        "successful_completion": True,
        "failed_ercp": False,
        "scout_film_status": "normal",
        "scout_film_optional_findings": "none",
        "scout_film_free_text": "none",
        "scope_advancement_difficulty": "without_difficulty",
        "scope_advancement_difficulty_reason": "none",
        "scope_advance_difficulty": "without_difficulty",
        "upper_gi_examination": "limited",
        "upper_gi_examination_extent": "limited",
        "upper_gi_findings": "normal",
        "biliary_stent_placed": True,
        "plastic_stent_details": "10 Fr x 5 cm plastic biliary stent",
        "pancreatic_stent_placed": False,
        "estimated_blood_loss": -1,
        "impressions": ["ERCP performed; stones removed."],
    }

    # Ensure all fields declared in the fields config are present in test_data
    for group in config.get('field_groups', {}).values():
        for f in group.get('fields', []):
            fname = f.get('name')
            if fname and fname not in test_data:
                # set booleans to False by default if type is boolean, else None
                ftype = f.get('type', 'string')
                if ftype == 'boolean':
                    test_data[fname] = False
                elif ftype in ('integer', 'number', 'float'):
                    test_data[fname] = -1
                else:
                    test_data[fname] = None
    
    # Render using the generated base template and write to txt
    temp_yaml_path = base_output
    rendered = build_report_sections(str(temp_yaml_path), test_data)
    output_path = _here/"demo_ercp_report.txt"
    parts = []
    for section_name, content in rendered.items():
        parts.append(f"== {section_name.upper()} ==\n\n{content.strip()}\n")
    output_text = "\n".join(parts).strip() + "\n"
    with open(output_path, "w") as out_f:
        out_f.write(output_text)
    print(f"\nRendered test procedure note written to: {output_path}\n")

if __name__ == "__main__":
    ''' To run demo: python templating/demo_ercp_yaml_pipeline.py'''
    demo()

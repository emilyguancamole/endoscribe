"""
Generate LLM prompts and report templates from field definitions.
Reads yaml file and generates:
1. LLM extraction prompt 
2. Report base template
3. Build report sections from templates and data

Usage:
    python reporting/generate_from_fields.py prompts/ercp/fields.yaml
"""

import yaml
import os
import sys
from typing import Dict, Any
from collections import OrderedDict
from copy import deepcopy

def merge_configs(base: dict, addon: dict) -> dict:
    """
    Deep merge two configuration dictionaries by:
        - field_groups: merged by group name, add-on config takes precedence
        - meta: merged, addon values override base
        - system_instructions: addon replaces base
        - Other top-level keys: addon replaces base
    """
    result = deepcopy(base)

    for key, value in addon.items():
        if key == '$extends':
            continue
        elif key == 'field_groups':
            if 'field_groups' not in result:
                result['field_groups'] = deepcopy(value)
            else:
                insert_after_hint = addon.get('meta', {}).get('insert_after')
                extraction_group = addon.get('meta', {}).get('extraction_group')
                result['field_groups'] = merge_field_groups_with_position(
                    result['field_groups'], 
                    value, 
                    insert_after_hint,
                    extraction_group
                )
        elif key == 'meta':
            if 'meta' not in result:
                result['meta'] = {}
            result['meta'].update(value)

        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge_dicts(result[key], value)

        else:
            result[key] = deepcopy(value)

    return result

def deep_merge_field_groups(base: dict, override: dict) -> dict:
    """Deep-merge field_groups: add fields and override templates."""
    result = deepcopy(base)
    
    for group_name, group_data in override.items():
        if group_name in result:
            # Merge this group
            for key, value in group_data.items():
                if key == 'fields' and 'fields' in result[group_name]:
                    # Append new fields to existing
                    existing_field_names = {f['name'] for f in result[group_name]['fields']}
                    for field in value:
                        if field['name'] not in existing_field_names:
                            result[group_name]['fields'].append(field)
                else:
                    # Override other keys (template, description,)
                    result[group_name][key] = value
        else:
            # New group
            result[group_name] = deepcopy(group_data)
    
    return result


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. Child values override base; nested dicts merged recursively.
    Lists are replaced by default.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def merge_field_groups_with_position(
    base_groups: dict, 
    addon_groups: dict, 
    insert_after: str = None,
    extraction_group: str = None
) -> dict:
    """Merge field_groups with optional module-wide insert_after placement.

    - Existing groups are deep-merged
    - New groups are inserted after the specified anchor if insert_after
    - Groups without placement hint are appended at the end in addon order.
    - extraction_group from module metadata is added to each new field_group
    """
    # Handle empty/missing addon_groups
    if not addon_groups:
        return deepcopy(base_groups)
    
    # Merge any groups that already exist
    merged = deepcopy(base_groups)
    existing_keys = list(merged.keys())

    # Merge groups that exist in both
    for name, cfg in addon_groups.items():
        if name in merged:
            merged = deep_merge_field_groups(merged, {name: cfg})

    # Collect new groups
    new_groups = []
    for name, cfg in addon_groups.items():
        if name not in merged:
            cfg_copy = deepcopy(cfg)
            if extraction_group:
                cfg_copy['extraction_group'] = extraction_group
            new_groups.append((name, cfg_copy))
    if not new_groups:
        return merged

    result = OrderedDict()
    # Rebuild ordered dict with new groups inserted
    for base_key in existing_keys:
        result[base_key] = merged[base_key]
        # If this is the anchor, insert all new groups after it
        if insert_after and base_key == insert_after:
            for name, cfg in new_groups:
                result[name] = cfg
    # If no anchor, append at end
    if not insert_after or insert_after not in existing_keys:
        for name, cfg in new_groups:
            if name not in result:
                result[name] = cfg

    return dict(result)

def load_procedure_config(config_path: str) -> Dict[str, Any]:
    """Load a procedure config that may extend another via "$extends": "path/to/base.yaml".
    Child overrides are deep-merged over base. Paths can be relative to the current file.
    Alias for load_fields_config to support legacy code.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    config = load_fields_config(config_path)
    return config

def load_fields_config(fields_yaml_path: str) -> dict:
    """Load the field definitions YAML with $extends support."""
    with open(fields_yaml_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    if '$extends' in config:
        base_rel_path = config['$extends']
        base_path = resolve_path(fields_yaml_path, base_rel_path)
        # If not found, try one level up (module subtypes)
        if not os.path.exists(base_path):
            alt = os.path.normpath(os.path.join(os.path.dirname(fields_yaml_path), '..', base_rel_path))
            if os.path.exists(alt):
                base_path = alt

        if os.path.exists(base_path):
            base_config = load_fields_config(base_path)
            override = {k: v for k, v in config.items() if k != '$extends'}
            merged = merge_configs(base_config, override)
            return merged
        else:
            print(f"Warning: base procedure yaml file not found for $extends='{base_rel_path}' (found {base_path})")
    
    return config

def resolve_path(base_path: str, rel_or_abs: str) -> str:
    """Resolve relative or absolute path"""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(os.path.dirname(base_path), rel_or_abs))


def build_report_sections(config_fp: str, data: Dict[str, Any]) -> Dict[str, str]:
    """Load a procedure config and render its templates with provided data.
    Returns a map of section_name -> rendered text.
    
    Args:
        config_fp: Path to procedure YAML config file
        data: Dictionary of field data to render into templates
    
    Returns:
        Dict mapping section names to rendered text
    """
    from .renderer import render_sections
    
    cfg = load_procedure_config(config_fp)
    templates = cfg.get("templates", {})
    return render_sections(templates, data)


def generate_prompt_text(config: dict) -> str:
    """Generate the LLM extraction prompt from field definitions."""
    lines = []
    
    # Add system instructions
    if 'system_instructions' in config:
        lines.append(config['system_instructions'].strip())
        lines.append("")
    
    # Add field definitions
    lines.append("The entities you must extract are listed below. Follow the instructions.")
    lines.append("###ENTITIES############################")
    
    # Iterate through field groups
    for _, group_data in config.get('field_groups', {}).items():
        if 'fields' in group_data:
            for field in group_data['fields']:
                field_name = field['name']
                instruction = field.get('prompt_instruction', '')
                lines.append(f"{field_name}: {instruction}")
                
    lines.append("\n###FURTHER INSTRUCTIONS############################")
    lines.append("The transcript may have spelling or transcription mistakes; use your domain expertise of endoscopy to correct them before recording an answer. If there are self-corrections for a finding later in the transcript, include only the most recent correct information. For example, if the transcript has: 'There were no masses or polyps found. But I'm checking again and I now see a 6 mm polyp in the transverse colon.' The second statement is a correction. The actual polyps size in mm you should record is 6.")
    lines.append("Follow the instructions for each field carefully; record none/unknown for unknown/NA fields.")
    lines.append("Return the result as a JSON file. Do not return any additional comments or explanation.")
    
    return "\n".join(lines)


def generate_base_yaml(config: dict, procedure_meta: dict) -> str:
    """Generate the base.yaml template from field definitions (e.g. from fields_base.yaml + subtypes)
    This new base.yaml is for Drafter.

    procedure_meta: dict with keys: procedure_group, procedure_type
    """
    if procedure_meta is None:
        print(f"Error: procedure_meta is required to generate base.yaml")
        sys.exit(1)
    
    # Generate prompt file reference based on procedure type
    proc_group = procedure_meta.get('procedure_group', 'ercp')
    proc_type = procedure_meta.get('procedure_type', proc_group)
    prompt_file = f'prompts/{proc_group}/generated_{proc_type}_prompt.txt'
    
    base = {
        'meta': procedure_meta,
        'prompts': {
            'definition_file': prompt_file
        },
        'templates': {}
    }
    
    # Group templates by report_section. Allow multiple groups to share the same
    # report_subsection by collecting templates in lists instead of overwriting.
    section_templates = {}

    for group_name, group_data in config.get('field_groups', {}).items():
        section = group_data.get('report_section')
        if section and 'template' in group_data:
            subsection = group_data.get('report_subsection')

            # Ensure section entry is a dict mapping subsection -> list_of_templates
            if section not in section_templates:
                section_templates[section] = {}

            if subsection:
                section_templates[section].setdefault(subsection, []).append(group_data['template'].strip())
            else:
                # Use special '_main' key for top-level section templates
                section_templates[section].setdefault('_main', []).append(group_data['template'].strip())
    
    # Flatten into templates
    for section, content in section_templates.items():
        # content is a dict mapping subsection -> list_of_templates
        parts = []
        # add main templates first (preserve order)
        if '_main' in content:
            parts.extend(content['_main'])
        # then add subsections in insertion order
        for subsection, templates_list in content.items():
            if subsection == '_main':
                continue
            parts.extend(templates_list)

        base['templates'][section] = '\n\n'.join(parts)
    
    # Convert to YAML string
    return yaml.dump(base, sort_keys=False, allow_unicode=True, default_flow_style=False)


def generate_pydantic_model(config: dict, model_name: str = "ERCPData") -> str:
    """Generate a Pydantic model from field definitions."""
    lines = [
        "from typing import List, Optional, Union",
        "from pydantic import BaseModel, field_validator",
        "",
        f"class {model_name}(BaseModel):",
    ]
    
    # Collect all fields and field names by type for validators
    fields = []
    boolean_fields = []
    integer_fields = []
    float_fields = []
    
    for _, group_data in config.get('field_groups', {}).items():
        if 'fields' in group_data:
            for field in group_data['fields']:
                field_name = field['name']
                field_type = field.get('type', 'string')

                type_map = {
                    'string': 'str',
                    'boolean': 'bool',
                    'integer': 'int',
                    'float': 'float',
                    'number': 'float',
                    'enum': 'str'
                }
                if field_type == 'list':
                    py_type = 'List[str]'
                else:
                    py_type = type_map.get(field_type, 'str')
                    # Track fields by type for validators
                    if field_type == 'boolean':
                        boolean_fields.append(field_name)
                    elif field_type == 'integer':
                        integer_fields.append(field_name)
                    elif field_type in ('float', 'number'):
                        float_fields.append(field_name)

                required = field.get('required', False)
                if required:
                    lines.append(f"    {field_name}: {py_type}")
                else:
                    lines.append(f"    {field_name}: Optional[{py_type}] = None")

                fields.append(field_name)
    
    # Add validator for XX fields to handle "unknown" string
    # Integer
    if integer_fields:
        lines.append("")
        lines.append("    @field_validator(" + ", ".join(f"'{f}'" for f in integer_fields) + ", mode='before')")
        lines.append("    @classmethod")
        lines.append("    def coerce_integer_with_unknown(cls, v):")
        lines.append("        \"\"\"Handle 'unknown' sentinel for integer fields.\"\"\"")
        lines.append("        if v is None:")
        lines.append("            return None")
        lines.append("        if isinstance(v, int):")
        lines.append("            return v")
        lines.append("        if isinstance(v, str):")
        lines.append("            v_lower = v.lower().strip()")
        lines.append("            # Treat unknown/none as None")
        lines.append("            if v_lower in ('unknown', 'none', 'n/a', 'na', '-1'):")
        lines.append("                return None")
        lines.append("            # Try to parse as integer")
        lines.append("            try:")
        lines.append("                return int(v)")
        lines.append("            except ValueError:")
        lines.append("                return None")
        lines.append("        return v")
    
    # Float
    if float_fields:
        lines.append("")
        lines.append("    @field_validator(" + ", ".join(f"'{f}'" for f in float_fields) + ", mode='before')")
        lines.append("    @classmethod")
        lines.append("    def coerce_float_with_unknown(cls, v):")
        lines.append("        \"\"\"Handle 'unknown' sentinel for float fields.\"\"\"")
        lines.append("        if v is None:")
        lines.append("            return None")
        lines.append("        if isinstance(v, (int, float)):")
        lines.append("            return float(v)")
        lines.append("        if isinstance(v, str):")
        lines.append("            v_lower = v.lower().strip()")
        lines.append("            # Treat unknown/none as None")
        lines.append("            if v_lower in ('unknown', 'none', 'n/a', 'na', '-1'):")
        lines.append("                return None")
        lines.append("            # Try to parse as float")
        lines.append("            try:")
        lines.append("                return float(v)")
        lines.append("            except ValueError:")
        lines.append("                return None")
        lines.append("        return v")
    
    # Boolean
    if boolean_fields:
        lines.append("")
        lines.append("    @field_validator(" + ", ".join(f"'{f}'" for f in boolean_fields) + ", mode='before')")
        lines.append("    @classmethod")
        lines.append("    def coerce_boolean_with_unknown(cls, v):")
        lines.append("        \"\"\"Handle 'unknown' sentinel for boolean fields and coerce truthy/falsey strings.\"\"\"")
        lines.append("        if v is None:")
        lines.append("            return None")
        lines.append("        if isinstance(v, bool):")
        lines.append("            return v")
        lines.append("        if isinstance(v, str):")
        lines.append("            v_lower = v.lower().strip()")
        lines.append("            # Treat unknown/none as None")
        lines.append("            if v_lower in ('unknown', 'none', 'n/a', 'na', '-1'):")
        lines.append("                return None")
        lines.append("            # Coerce truthy/falsey strings")
        lines.append("            if v_lower in ('true', 'yes', '1'):")
        lines.append("                return True")
        lines.append("            if v_lower in ('false', 'no', '0'):")
        lines.append("                return False")
        lines.append("        # Pass through as-is and let pydantic handle validation")
        lines.append("        return v")
    
    return "\n".join(lines)


def generate_single(fields_yaml_path: str, proc_type: str = None):
    """Generate artifacts from a single fields.yaml file."""
    print(f"Loading field definitions from: {fields_yaml_path}")
    config = load_fields_config(fields_yaml_path)
    
    # Determine procedure type and group from config or filename
    if proc_type is None:
        proc_type = config.get('meta', {}).get('procedure_type', 'unknown')
    proc_group = config.get('meta', {}).get('procedure_group', proc_type)
    meta = config.get('meta', {}) or {}
    is_subtype = False
    if 'module_id' in meta:
        is_subtype = True

    # Generate prompt
    prompt_text = generate_prompt_text(config)
    prompt_dir = os.path.join('prompts', proc_group)
    if is_subtype:
        prompt_dir = os.path.join(prompt_dir, 'subtypes')        

    os.makedirs(prompt_dir, exist_ok=True)
    prompt_output = os.path.join(prompt_dir, f'generated_{proc_type}_prompt.txt')
    with open(prompt_output, 'w') as f:
        f.write(prompt_text)
    print(f"  Generated prompt: {prompt_output}")
    
    # Generate yaml for Drafter
    procedure_meta = config.get('meta', {
        'procedure_group': proc_type,
        # 'title': 'Endoscopic Procedure'
    })
    base_yaml = generate_base_yaml(config, procedure_meta)
    proc_group = procedure_meta.get('procedure_group', 'ercp')
    base_output = os.path.join('drafters', 'procedures', proc_group, f'generated_{proc_type}.yaml')
    os.makedirs(os.path.dirname(base_output), exist_ok=True)
    with open(base_output, 'w') as f:
        f.write(base_yaml)
    print(f"  Generated base template: {base_output}")
    
    # Generate Pydantic model
    model_name = f"{proc_type.replace('_', ' ').title().replace(' ', '')}Data"
    model_code = generate_pydantic_model(config, model_name)
    # Write generated models - JUST for inspection
    model_output = os.path.join('models', f'generated_{proc_type}_model.py')
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    with open(model_output, 'w') as f:
        f.write(model_code)
    print(f"  Generated Pydantic model: {model_output}")


def main():
    """Test out the generation from fields.yaml and see the generated files"""
    if len(sys.argv) < 2:
        print("Usage: python templating/generate_from_fields.py <fields.yaml>")
        print("Example: python templating/generate_from_fields.py prompts/ercp/yaml/fields_cholangioscopy.yaml")
        print("\nOr all registered procedures: python templating/generate_from_fields.py --all")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        # Generate from procedure registry
        registry_path = 'prompts/procedure_registry.yaml'
        if not os.path.exists(registry_path):
            print(f"Error: Registry not found: {registry_path}")
            sys.exit(1)
        
        import yaml
        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f)
        
        print(f"Generating from procedure registry: {registry_path}\n")
        for proc_type, proc_info in registry.get('procedures', {}).items():
            fields_path = proc_info.get('fields_file')
            if fields_path and os.path.exists(fields_path):
                print(f"\n{'='*60}")
                print(f"Processing: {proc_type}")
                print(f"{'='*60}")
                generate_single(fields_path, proc_type)
            else:
                print(f"  Skipping {proc_type}: fields file not found")
        
        print("\n" + "="*60)
        print("All procedures generated!")
        return
    
    # Single file generation
    fields_yaml_path = sys.argv[1]
    if not os.path.exists(fields_yaml_path):
        print(f"Error: File not found: {fields_yaml_path}")
        sys.exit(1)
    
    generate_single(fields_yaml_path)


if __name__ == "__main__":
    main()

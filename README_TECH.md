# EndoScribe Technical Reference

# Quick Reference: FULL EndoScribe Pipeline
```bash
# Generate all artifacts from fields.yaml
python templating/generate_from_fields.py prompts/ercp/yaml/fields_base.yaml
  # Test the generated template with dummy data
  python templating/demo_ercp_yaml_pipeline.py

# Transcribe single file with Azure
python -m transcription.azure_transcribe --audio_file transcription/recordings/ercp/bdstone/bdstone03.m4a --procedure_type=ercp

# LLM Extraction
python main.py --procedure_type=ercp --transcripts_fp=azure_trs.csv --output_filename=azure_ext --files_to_process bdstone01

# Drafter
python drafter.py --procedure=ercp --pred_csv=azure_ext.csv --output_dir=drafters/results/ercp/templated --samples_to_process bdstone01 

# Reviewer
# TODO
```
-------------------------------------------------

# TECH DETAILS
# YAML Field-to-Report Pipeline

**ONE YAML FILE** → generates → **Prompt + Template + Model**
> fields.yaml  →  [generator]  →  prompt.txt (for llm) + base.yaml (for drafter) + model.py (pydantic model)


## Field Groups + Definitions
```yaml
field_groups:
  my_section:
    report_section: "section_name"
    template: "Text with {{ field_name }}."
    fields:
      - name: field_name
        type: string              # string, boolean, integer, enum
        required: true            # Pydantic won't allow None
        prompt_instruction: "..." # what LLM should extract
        enum_values: ["a", "b"]  
```

### Template Patterns
```yaml
# Skip unknown/none
{{ field | sku }}

# Conditional section
{% if field | sku %}Text here.{% endif %}

# Multi-choice
{% if status == 'normal' %}
  Normal.
{% elif status == 'abnormal' %}
  Abnormal: {{ findings }}.
{% endif %}

# Set variable
{% set var = field | default('') | sku %}
```

### Jinja Filters

| Filter | Use |
|--------|-----|
| `sku` | Skip "unknown"/"none" |
| `sent` | Add period (sentence) |
| `capfirst` | Capitalize first letter |
| `default_if_unknown` | Fallback value |
| `default_if_unknown_sentence` | Fallback as sentence, fallback presented as a properly capitalized sentence with punctuation. |
| `join_nonempty` | Join list, skip empty |


### Adding a Field
```yaml
# 1. Add to fields.yaml
fields:
  - name: new_field
    prompt_instruction: "Extract this info"

# 3. Regenerate
python reporting/generate_from_fields.py prompts/ercp/fields.yaml
```

### Adding a Subtype
described in [PROCEDURE_SUBTYPES.md](./PROCEDURE_SUBTYPES.md)

## Templating WORKFLOW
1. **Edit** `prompts/ercp/yaml/fields_base.yaml`
2. **Generate**: `python templating/generate_from_fields.py prompts/ercp/yaml/fields_base.yaml`
3. **Test**: Has dummy data (may need to update after yaml changes) so I can test the flow without LLM calls: `python templating/demo_ercp_yaml_pipeline.py`
4. **Review** generated files

### More Docs
- `reporting/README.md` - Template customization
- `reporting/demo_ercp_yaml_pipeline.py` - Working example full pipeline
- `reporting/PROCEDURE_SUBTYPES.md` - How to add new procedure subtypes


# Drafter
Drafter formats extracted data into clinical note drafts (`.docx`), using drafters/procedures/{proc_type}/generated_{proc}_base.yaml.


# Transcription


# LLM Extraction

# Reviewer

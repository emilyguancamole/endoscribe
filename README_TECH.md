# EndoScribe Technical Reference
last updated: 12/13/25

# Reference: FULL EndoScribe Pipeline
```bash
# Generate artifacts  from fields.yaml
python templating/generate_from_fields.py prompts/ercp/yaml/fields_base.yaml
  # This outputs: LLM prompt (generated_{proc}_prompt.txt), drafter model (generated_{proc}.yaml), data model (generated_{proc}_model.py))

  # Test the generated template with dummy data
  python templating/demo_ercp_yaml_pipeline.py

# Transcribe single file 
  python -m transcription.transcription_service --procedure_type=ercp --audio_file transcription/recordings/ercp/bdstone/bdstone07.m4a
    --service=[azure,whisperx]
    # defaults: --service=azure --save_filename=transcription/results/ercp/{service}_trs.csv
    

# LLM Extraction
python main.py --procedure_type=ercp --transcripts_fp=azure_trs.csv --output_filename=azure_ext --files_to_process bdstone01

# Drafter
python drafter.py --procedure=ercp --pred_csv=azure_ext.csv --output_dir=drafters/results/ercp/templated --samples_to_process bdstone01 

# Reviewer
# TODO
```

# TECH DETAILS
# YAML Field-to-Report Pipeline

**ONE YAML FILE** → generates → **Prompt + Template + Model**
> fields.yaml  →  [generator]  →  prompt.txt (for llm) + base.yaml (for drafter) + model.py (pydantic model)


### Field Groups + Definitions
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

# Make a paragraph without newlines (note: run replace('  ', ' ') repeatedly to catch double spaces)
{% set _section_name %}
text
more text
{% endset %}
{{ _section_name | replace('\n', ' ') | replace('  ', ' ')  | replace('  ', ' ') | trim}}
```

### Jinja Filters

| Filter | Use | Example |
|--------|-----|---------|
| `sku` | Skip "unknown"/"none" |
| `sent` | Add period (sentence) | {{ field | sent }} → "Field text."
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
For each drafted report the drafter saves a package JSON at {sample}_package.json with keys: 'sample_id', 'meta', 'original_transcript', 'extracted_fields', 'rendered_sections', 'rendered_note', 'docx_path', 'provenance'.
In `reviewer.py`:
- run_reviewer_on_package(package_fp, llm_handler=None, dry_run=False) reads the package JSON.
- Builds a system_msg instructing the LLM to "Return ONLY a single JSON object" following a schema (top-level fields: updated_fields, updated_sections, accept_rendered_note, final_note, deltas, confidence, warnings).
- Builds user_msg containing a JSON dump with sample_id, meta, original_transcript, extracted_fields_normalized, extracted_fields_raw, rendered_sections.
- calls `llm_handler.chat(...)` with `messages`.
- Uses `_extract_json(response_text)` to try to find a JSON substring in the LLM output. Loads JSON and runs _basic_validate_schema(parsed) to ensure minimal fields. If failure, writes raw response to `<package>_reviewer_raw.txt` and raises an error. Otherwise writes `<package>_reviewer.json`.

TODO re-draft the note?? how to integrate with drafter?
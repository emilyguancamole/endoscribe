# YAML Field-to-Report Pipeline - Quick Reference

**ONE YAML FILE** → generates → **Prompt + Template + Model**

```
fields.yaml  →  [generator]  →  prompt.txt + base.yaml + model.py
```

## Quick Commands

```bash
# Generate all artifacts from fields.yaml
python reporting/generate_from_fields.py prompts/ercp/fields.yaml

# View generated files
cat prompts/ercp/generated_prompt.txt
cat prompts/procedures/ercp/generated_base.yaml
cat prompts/ercp/generated_model.py

# Test the generated template with dummy data
python templating/demo_ercp_yaml_pipeline.py
```

### Minimal Field Group
```yaml
field_groups:
  my_section:
    report_section: "section_name"
    template: "Text with {{ field_name }}."
    fields:
      - name: field_name
        prompt_instruction: "What LLM should extract"
```

### Full Field Definition
```yaml
- name: field_name
  type: string              # string, boolean, integer, enum
  required: true            # Pydantic won't allow None
  prompt_instruction: "..."
  enum_values: ["a", "b"]   # For type: enum
```

### Common Template Patterns
```yaml
# Skip unknown/none
{{ field | skip_unknown }}

# Conditional section
{% if field | skip_unknown %}Text here.{% endif %}

# Multi-choice
{% if status == 'normal' %}
  Normal.
{% elif status == 'abnormal' %}
  Abnormal: {{ findings }}.
{% endif %}

# Set variable
{% set var = field | default('') | skip_unknown %}
```

## WORKFLOW

1. **Edit** `prompts/ercp/yaml/fields_base.yaml`
2. **Generate**: `python templating/generate_from_fields.py prompts/ercp/yaml/fields_base.yaml`
3. **Test**: Has dummy data (may need to update after yaml changes) so I can test the flow without LLM calls: `python templating/demo_ercp_yaml_pipeline.py`
4. **Review** generated files
5. **Deploy**: Copy to production when satisfied

## Jinja Filters

| Filter | Use |
|--------|-----|
| `skip_unknown` | Omit "unknown"/"none" |
| `sentence` | Add period |
| `capfirst` | Capitalize first letter |
| `default_if_unknown` | Fallback value |
| `default_if_unknown_sentence` | Fallback as sentence, fallback presented as a properly capitalized sentence with punctuation. |
| `join_nonempty` | Join list, skip empty |


## Adding a Field
```yaml
# 1. Add to fields.yaml
fields:
  - name: new_field
    prompt_instruction: "Extract this info"

# 3. Regenerate
python reporting/generate_from_fields.py prompts/ercp/fields.yaml
```

## Adding a Subtype

described in [PROCEDURE_SUBTYPES.md](./PROCEDURE_SUBTYPES.md)

## Issues

**Field not rendering?**
- Check spelling (case-sensitive)
- Add `| default('')` before filters

## More Docs

- `reporting/README.md` - Template customization
- `reporting/demo_ercp_yaml_pipeline.py` - Working example full pipeline
- `reporting/PROCEDURE_SUBTYPES.md` - How to add new procedure subtypes
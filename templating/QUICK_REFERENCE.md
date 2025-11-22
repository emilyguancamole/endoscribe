# YAML Field-to-Report Pipeline - Quick Reference

## 🎯 Core Concept

**ONE YAML FILE** → generates → **Prompt + Template + Model**

```
fields.yaml  →  [generator]  →  prompt.txt + base.yaml + model.py
```

## ⚡ Quick Commands

```bash
# Generate all artifacts from fields.yaml
python reporting/generate_from_fields.py prompts/ercp/fields.yaml

# Test the generated template
python reporting/demo_yaml_pipeline.py

# View generated files
cat prompts/ercp/generated_prompt.txt
cat prompts/procedures/ercp/generated_base.yaml
cat prompts/ercp/generated_model.py
```

## 📝 fields.yaml Cheat Sheet

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

## 🔧 Workflow

1. **Edit** `prompts/ercp/fields.yaml`
2. **Generate**: `python reporting/generate_from_fields.py prompts/ercp/fields.yaml`
3. **Test**: `python reporting/demo_yaml_pipeline.py`
4. **Review** generated files
5. **Deploy**: Copy to production when satisfied

## 📂 File Map

| Purpose | Source | Generated | Production |
|---------|--------|-----------|------------|
| Field defs | `prompts/ercp/fields.yaml` | — | — |
| LLM prompt | — | `generated_prompt.txt` | `ercp_def_1.txt` |
| Template | — | `generated_base.yaml` | `base.yaml` |
| Data model | — | `generated_model.py` | merge into `data_models.py` |

## ✨ Jinja Filters

| Filter | Use |
|--------|-----|
| `skip_unknown` | Omit "unknown"/"none" |
| `sentence` | Add period |
| `capfirst` | Capitalize first letter |
| `default_if_unknown` | Fallback value |
| `join_nonempty` | Join list, skip empty |

## 🚀 Adding a Field

```yaml
# 1. Add to fields.yaml
fields:
  - name: new_field
    prompt_instruction: "Extract this info"

# 2. Add to template
template: "{{ new_field | skip_unknown }}"

# 3. Regenerate
python reporting/generate_from_fields.py prompts/ercp/fields.yaml
```

## 🎨 Report Sections

Standard section names:
- `indications`
- `history`
- `description_of_procedure`
- `findings`
- `ercp_quality_metrics`
- `impressions`
- `recommendations`

Use `report_subsection` for nested content.

## 💡 Pro Tips

- **Test first**: Use demo before deploying
- **Version control**: Keep fields.yaml in git
- **Incremental**: Add one field at a time
- **Comments**: Document complex logic in YAML
- **DRY**: Put common text in base, override in variants

## 🐛 Common Issues

**Field not rendering?**
- Check spelling (case-sensitive)
- Add `| default('')` before filters

**Template syntax error?**
- Validate Jinja with demo
- Check for unmatched `{% %}`

**Generated prompt incomplete?**
- Ensure field is under `fields:` array
- Check YAML indentation

## 📚 Full Docs

- `reporting/YAML_DRIVEN_PIPELINE.md` - Complete guide
- `reporting/README.md` - Template customization
- `reporting/demo_yaml_pipeline.py` - Working example

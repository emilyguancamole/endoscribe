# Procedure Subtypes with Inheritance

## 🎯 Problem Solved

You have **multiple procedure subtypes** (e.g., ERCP cholangioscopy, ERCP normal follow-up, ERCP gastrogastrostomy) that:
- Share common fields (patient demographics, history, scope type)
- Have subtype-specific fields (cholangioscopy findings, stent management, altered anatomy)
- Need separate LLM prompts and report templates

**Solution**: Hierarchical field definitions with `$extends` inheritance.

## 📁 File Structure

```
prompts/
├── procedure_registry.yaml          # Central registry of all procedure types
└── ercp/
    ├── fields_base.yaml             # Common ERCP fields (23 fields)
    ├── fields_cholangioscopy.yaml   # + 6 cholangioscopy fields = 29 total
    ├── fields_nfu.yaml              # + 11 follow-up fields = 34 total
    └── fields_gastrogastrostomy.yaml # + 6 anatomy fields = 29 total
```

## 🔗 How Inheritance Works

### Base File (fields_base.yaml)
```yaml
# Common to ALL ERCPs
meta:
  procedure_group: ercp
  title: Endoscopic Retrograde Cholangiopancreatography

field_groups:
  patient_demographics:
    fields:
      - name: age
      - name: sex
      - name: chief_complaints
  # ... more common groups
```

### Subtype File (fields_cholangioscopy.yaml)
```yaml
$extends: ./fields_base.yaml  # Inherit all base fields

# Override metadata
meta:
  procedure_type: ercp_cholangioscopy
  title: ERCP with Cholangioscopy

# Override specific templates
field_groups:
  patient_demographics:
    template: "{{ age }} year old {{ sex }} for ERCP with Cholangioscopy..."
  
  # Add NEW field groups specific to this subtype
  cholangioscopy_findings:
    fields:
      - name: spyglass_used
      - name: bile_duct_visualization
      # ... cholangioscopy-specific fields
```

## ⚙️ Generator Behavior

1. **Loads base** (`fields_base.yaml`) first
2. **Merges subtype** on top:
   - New field_groups are added
   - Existing field_groups are updated (templates overridden, fields appended)
   - Meta information is merged
3. **Generates three files**:
   - LLM prompt with ALL fields (base + subtype)
   - Report template with ALL sections
   - Pydantic model with ALL fields

## 🚀 Usage

### Generate All Subtypes
```bash
python templating/generate_from_fields.py --all
```

This reads `prompts/procedure_registry.yaml` and generates artifacts for every registered procedure.

### Generate Single Subtype
```bash
python templating/generate_from_fields.py prompts/ercp/fields_cholangioscopy.yaml
```

### Generated Files
For `ercp_cholangioscopy`:
- `prompts/ercp/generated_ercp_cholangioscopy_prompt.txt` (29 fields)
- `drafters/procedures/ercp/generated_ercp_cholangioscopy.yaml` (template)
- `prompts/ercp/generated_ercp_cholangioscopy_model.py` (Pydantic)

## 📝 Adding a New Subtype

### Step 1: Create Field Definition

```yaml
# prompts/ercp/fields_stone_extraction.yaml
$extends: ./fields_base.yaml

meta:
  procedure_type: ercp_stone_extraction
  title: ERCP for Stone Extraction

field_groups:
  # Add stone-specific fields
  stone_extraction:
    report_section: "stone_extraction_details"
    template: |
      {% if stone_present %}
      Bile duct stones were identified.
      {% if stone_size | skip_unknown %}Stone size: {{ stone_size }}.{% endif %}
      {% if extraction_method | skip_unknown %}
      Extraction method: {{ extraction_method }}.
      {% endif %}
      {% endif %}
    fields:
      - name: stone_present
        type: boolean
        prompt_instruction: "Were bile duct stones identified? yes/no"
      
      - name: stone_size
        type: string
        prompt_instruction: "Size of largest stone if present, e.g. '8mm'"
      
      - name: extraction_method
        type: string
        prompt_instruction: "Method used for stone extraction, e.g. 'basket', 'balloon', 'lithotripsy'"
```

### Step 2: Register in procedure_registry.yaml

```yaml
procedures:
  ercp_stone_extraction:
    fields_file: prompts/ercp/fields_stone_extraction.yaml
    config_file: drafters/procedures/ercp/config_stone_extraction.yaml
    description: "ERCP for bile duct stone extraction"
```

### Step 3: Generate

```bash
python templating/generate_from_fields.py --all
```

### Step 4: Create Variant Config

```yaml
# drafters/procedures/ercp/stone_extraction.yaml
$extends: ./generated_ercp_stone_extraction.yaml
# Add any final customizations if needed
```

## 🎨 Customization Strategies

### Override a Template
```yaml
# In subtype file
field_groups:
  patient_demographics:
    template: "Custom template for this subtype only..."
```

### Add Fields to Existing Group
```yaml
# In subtype file
field_groups:
  patient_history:  # Already exists in base
    fields:
      - name: specific_risk_factor
        prompt_instruction: "Subtype-specific risk factor"
```

### Add New Section
```yaml
# In subtype file
field_groups:
  new_section_name:
    report_section: "custom_findings"
    template: "New content here..."
    fields:
      - name: custom_field
```

## 🌐 Extending to Other Procedure Types

### EUS Example

```yaml
# prompts/eus/fields_base.yaml
meta:
  procedure_group: eus
  title: Endoscopic Ultrasound

field_groups:
  patient_demographics:
    template: "{{ age }} year old {{ sex }} for EUS..."
  # EUS-specific common fields
```

```yaml
# prompts/eus/fields_fna.yaml
$extends: ./fields_base.yaml

meta:
  procedure_type: eus_fna

field_groups:
  fna_details:
    fields:
      - name: needle_size
      - name: passes_performed
```

### Register in procedure_registry.yaml
```yaml
procedures:
  eus_fna:
    fields_file: prompts/eus/fields_fna.yaml
    config_file: drafters/procedures/eus/config_fna.yaml
    description: "EUS with fine needle aspiration"
```

## 📊 Field Count Summary

| Procedure Type | Base Fields | Added Fields | Total |
|---------------|-------------|--------------|-------|
| ERCP Base | 23 | — | 23 |
| ERCP Cholangioscopy | 23 | 6 | 29 |
| ERCP Normal Follow-up | 23 | 11 | 34 |
| ERCP Gastrogastrostomy | 23 | 6 | 29 |

## 🔍 How Drafter Selects Subtype

### Option 1: From Data (Recommended)
```python
# In procedures.csv, add a column: procedure_subtype
# e.g., "ercp_cholangioscopy", "ercp_normal_follow_up"

drafter = ERCPDrafter(sample, pred_df, patients_df, procedures_df)
drafter.procedure_variant = procedures_df.loc[sample].get('procedure_subtype', 'ercp_cholangioscopy')
doc = drafter.draft_doc()
```

### Option 2: Explicit Selection
```python
drafter = ERCPDrafter(sample, pred_df, patients_df, procedures_df)
drafter.procedure_variant = "ercp_gastrogastrostomy"
doc = drafter.draft_doc()
```

### Option 3: Auto-detect from Extracted Fields
```python
# In drafter
def select_variant(self):
    if self.sample_df.get('spyglass_used'):
        return 'ercp_cholangioscopy'
    elif self.sample_df.get('prior_stent_present'):
        return 'ercp_nfu
    elif self.sample_df.get('gg_anastomosis_location'):
        return 'ercp_gastrogastrostomy'
    return 'ercp_cholangioscopy'  # default
```

## 💡 Best Practices

1. **Keep base minimal** - Only truly universal fields
2. **Specific subtypes** - Create subtypes for distinct clinical scenarios
3. **Test inheritance** - Generate and review before deploying
4. **Document differences** - Add description to procedure_registry.yaml
5. **Version control** - Track both base and subtype files in git

## 🐛 Troubleshooting

**Fields not inherited?**
- Check `$extends` path is correct (relative to subtype file)
- Ensure base file exists and is valid YAML

**Template not overriding?**
- Match field_group name exactly (case-sensitive)
- Provide complete template (not partial)

**Too many/few fields?**
- Use `grep "^[a-z_]*:" generated_prompt.txt | wc -l` to count
- Check field_groups merge logic

## 📚 Related Docs

- `YAML_DRIVEN_PIPELINE.md` - Core concepts
- `QUICK_REFERENCE.md` - Command cheat sheet
- `procedure_registry.yaml` - All registered procedures

## ✨ Summary

**Before**: Maintain separate files for each subtype (error-prone, tedious)

**Now**: 
- Define base once (`fields_base.yaml`)
- Extend with `$extends` for each subtype
- Generate all artifacts with one command
- Always in sync, easy to maintain

🎉 **Subtypes supported with full inheritance!**

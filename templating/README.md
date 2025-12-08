# EndoScribe Templating System

**Smart, extensible report generation for endoscopy procedures with LLM field extraction and hierarchical templates.**

## 🎯 What Problem Does This Solve?

You have **multiple procedure types** (ERCP, EUS, EGD, Colonoscopy) with:
- **Shared data patterns** (patient demographics, procedure details)
- **Type-specific sections** (ERCP cholangioscopy findings, EUS FNA results)
- **LLM-extracted fields** that should appear conditionally in reports
- **Subtypes within categories** (ERCP cholangioscopy vs. ERCP follow-up)

**Old Way**: Manually maintain separate prompts, templates, and models → error-prone, inconsistent

**New Way**: Single source of truth (`fields.yaml`) → auto-generate prompts + templates + models with inheritance

## 🚀 Quick Start

### Generate All Procedure Artifacts
```bash
# Generate prompts, templates, and models for ALL registered procedures
python templating/generate_from_fields.py --all
```

### Generate Single Procedure
```bash
python templating/generate_from_fields.py prompts/ercp/fields_cholangioscopy.yaml
```

### Use in Drafter
```python
from templating.drafter_engine import build_report_sections

# Load variant-specific template and render with data
sections = build_report_sections(
    "drafters/procedures/ercp/cholangioscopy.yaml",
    {
        "age": 65,
        "sex": "male",
        "chief_complaints": "abdominal pain",
        "spyglass_used": True,
        "bile_duct_visualization": "adequate",
        # ... more fields
    }
)

# sections = {'history': '...', 'description_of_procedure': '...'}
```

## 📁 Directory Structure

```
templating/
├── README.md                        # This file - System overview
├── YAML_DRIVEN_PIPELINE.md          # Complete field→report pipeline docs
├── PROCEDURE_SUBTYPES.md            # Subtype inheritance guide
├── QUICK_REFERENCE.md               # Command cheat sheet
├── config_loader.py                 # YAML loader with $extends
├── renderer.py                      # Jinja2 engine with custom filters
├── drafter_engine.py                # High-level API
├── generate_from_fields.py          # Generator script
└── demo_*.py                        # Test/demo scripts

prompts/
├── procedure_registry.yaml          # Central registry of all procedures
└── ercp/
    ├── fields_base.yaml             # Base ERCP fields (23 fields)
    ├── fields_cholangioscopy.yaml   # Extends base + 6 fields
    ├── fields_nfu.yaml # Extends base + 11 fields
    ├── fields_gastrogastrostomy.yaml # Extends base + 6 fields
    └── generated_*.txt/.yaml/.py    # Auto-generated artifacts

drafters/procedures/
├── index.yaml                       # Procedure variant registry
└── ercp/
    ├── base.yaml                    # Base ERCP template
    ├── cholangioscopy.yaml          # Variant extending base
    └── generated_*.yaml             # Auto-generated templates
```

## 🔧 Core Components

### 1. Config Loader (`config_loader.py`)
Loads YAML configuration files with support for `$extends` inheritance.

```python
from templating.config_loader import load_procedure_config

config = load_procedure_config("drafters/procedures/ercp/cholangioscopy.yaml")
```

**Features:**
- Recursive `$extends` resolution (relative paths)
- Deep dictionary merging
- List append behavior for arrays

### 2. Renderer (`renderer.py`)
Renders Jinja2 templates with custom filters for handling unknown/none values.

```python
from templating.renderer import build_env, render_sections

env = build_env()
sections = render_sections(env, config, data_dict)
```

**Custom Filters:**
- `sku` - Skip unknown; return "" 
- `sentence` - Add trailing period if missing
- `capfirst` - Uppercase first letter only
- `default_if_unknown` - Provide fallback for unknown values
- `join_nonempty` - Join list items, skipping empty strings

### 3. Drafter Engine (`drafter_engine.py`)
High-level API combining config loading and rendering.

```python
from templating.drafter_engine import build_report_sections

sections = build_report_sections("path/to/config.yaml", data_dict)
```

**Usage in Drafter:**
```python
class ERCPDrafter(BaseDrafter):
    PROCEDURE_VARIANT = "ercp_cholangioscopy"  # Can be changed per instance
    
    def draft_doc(self):
        # Load variant-specific config
        index = load_yaml("drafters/procedures/index.yaml")
        config_fp = index['procedures'][self.PROCEDURE_VARIANT]
        
        # Build data dict from dataframes
        data = {
            "age": self.sample_df.get('age'),
            "sex": self.sample_df.get('sex'),
            # ... more fields
        }
        
        # Render sections
        sections = build_report_sections(config_fp, data)
        
        # Insert into DOCX
        self.add_heading("History")
        self.doc.add_paragraph(sections['history'])
```

### 4. Generator (`generate_from_fields.py`)
Converts `fields.yaml` → prompt + template + Pydantic model.

```bash
# Single procedure
python templating/generate_from_fields.py prompts/ercp/fields_cholangioscopy.yaml

# All registered procedures
python templating/generate_from_fields.py --all
```

**Outputs:**
- `generated_{type}_prompt.txt` - LLM extraction instructions
- `generated_{type}.yaml` - Report template sections
- `generated_{type}_model.py` - Pydantic validation model

## 📚 Documentation

### For New Users
Start with **QUICK_REFERENCE.md** for command cheatsheet and common tasks.

### For Field Definition Authors
Read **YAML_DRIVEN_PIPELINE.md** to understand:
- Field YAML schema
- How fields become prompts and templates
- Jinja template syntax
- Testing and validation

### For Subtype Management
Read **PROCEDURE_SUBTYPES.md** to learn:
- How `$extends` inheritance works
- Creating base + variant field files
- Field count management
- Procedure registry usage

## 🎨 Workflow: Adding a New Procedure Subtype

### Step 1: Create Field Definition
```yaml
# prompts/ercp/fields_mysubtype.yaml
$extends: ./fields_base.yaml

meta:
  procedure_type: ercp_mysubtype
  title: ERCP My Subtype

field_groups:
  my_custom_section:
    report_section: "custom_findings"
    template: |
      {% if custom_field | sku %}
      Custom finding: {{ custom_field | sent }}
      {% endif %}
    fields:
      - name: custom_field
        type: string
        prompt_instruction: "Extract custom field information"
```

### Step 2: Register in `procedure_registry.yaml`
```yaml
procedures:
  ercp_mysubtype:
    fields_file: prompts/ercp/fields_mysubtype.yaml
    config_file: drafters/procedures/ercp/mysubtype.yaml
    description: "ERCP with custom subtype"
```

### Step 3: Generate Artifacts
```bash
python templating/generate_from_fields.py --all
```

### Step 4: Review Generated Files
```bash
# Check prompt
cat prompts/ercp/generated_ercp_mysubtype_prompt.txt

# Check template
cat drafters/procedures/ercp/generated_ercp_mysubtype.yaml

# Check model
cat prompts/ercp/generated_ercp_mysubtype_model.py
```

### Step 5: Customize if Needed
```yaml
# drafters/procedures/ercp/mysubtype.yaml
$extends: ./generated_ercp_mysubtype.yaml

# Add manual customizations here
# e.g., override specific templates
```

### Step 6: Use in Drafter
```python
drafter = ERCPDrafter(sample, pred_df, patients_df, procedures_df)
drafter.PROCEDURE_VARIANT = "ercp_mysubtype"
doc = drafter.draft_doc()
```

## 🔗 Key Concepts

### Inheritance with `$extends`
```yaml
# Child inherits ALL from parent, can override/extend
$extends: ./parent.yaml

# New fields are added
# Existing fields are overridden (if same key)
# Lists are appended (not replaced)
```

### Conditional Rendering
```jinja
{% if field | sku %}
This only appears if field is NOT unknown/none/n/a
{% endif %}

{{ field | default_if_unknown("Default text") }}
```

### Report Sections
Each `field_group` maps to a `report_section`:
- `history` → Patient History section
- `description_of_procedure` → Description of Procedure section
- `findings` → Findings section
- Custom sections supported

### Procedure Registry
Central file (`prompts/procedure_registry.yaml`) lists all procedure types:
```yaml
procedures:
  ercp_cholangioscopy:
    fields_file: prompts/ercp/fields_cholangioscopy.yaml
    config_file: drafters/procedures/ercp/cholangioscopy.yaml
    description: "ERCP with cholangioscopy"
```

## 🧪 Testing

### Test Template Rendering
```bash
python templating/demo_render.py
```

### Test Full Pipeline
```bash
python templating/demo_yaml_pipeline.py
```

### Validate Field Definitions
```bash
# Generate and check for errors
python templating/generate_from_fields.py prompts/ercp/fields_cholangioscopy.yaml

# Check field count
grep "^[a-z_]*:" prompts/ercp/generated_ercp_cholangioscopy_prompt.txt | wc -l
```

## 🐛 Troubleshooting

### Fields Not Showing in Report
**Cause**: Field value is "unknown", "none", or "n/a"  
**Fix**: Check LLM extraction, or use `{{ field | default_if_unknown("fallback") }}`

### Template Override Not Working
**Cause**: Empty string override blocks inheritance  
**Fix**: Use `null` or omit the key entirely

### $extends Path Error
**Cause**: Relative path incorrect  
**Fix**: Use `./` for same directory, `../` for parent directory

### Generated Files in Wrong Location
**Cause**: Incorrect `procedure_group` in meta  
**Fix**: Ensure `meta.procedure_group` matches folder name (ercp, eus, etc.)

## 📊 Current Status

### Supported Procedures
| Procedure | Base Fields | Subtypes | Status |
|-----------|-------------|----------|--------|
| ERCP | 23 | 3 (cholangioscopy, follow-up, GG) | ✅ Complete |
| EUS | — | — | 🚧 TODO |
| EGD | — | — | 🚧 TODO |
| Colonoscopy | — | — | 🚧 TODO |

### ERCP Subtypes
| Subtype | Added Fields | Total Fields | Description |
|---------|--------------|--------------|-------------|
| Cholangioscopy | 6 | 29 | SpyGlass visualization |
| Normal Follow-up | 11 | 34 | Stent management |
| Gastrogastrostomy | 6 | 29 | Altered anatomy |

## 🎯 Next Steps

1. **Review generated artifacts** - Check prompts, templates, models for accuracy
2. **Test with real data** - Run drafters with actual patient data
3. **Deploy to production** - Copy from `generated_*` to production files
4. **Extend to other procedures** - Create EUS, EGD, Colonoscopy field definitions
5. **Add external procedure packs** - Implement search path support

## 📖 Additional Resources

- See `YAML_DRIVEN_PIPELINE.md` for complete field→report workflow
- See `PROCEDURE_SUBTYPES.md` for inheritance patterns
- See `QUICK_REFERENCE.md` for command reference
- See demos (`demo_*.py`) for working examples

---

**Built with ❤️ for flexible, maintainable medical report generation**

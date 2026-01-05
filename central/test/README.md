## Setup Instructions
Install Miniforge, which includes conda and mamba. We use conda (or mamba) to create an isolated environment for the project.
```
# for mac:
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
```
Verify installation by restarting terminal and running:
```
mamba --version
```

Create conda environment (e.g. named 'myenv') and install packages:

```bash
mamba create --name myenv python=3.10
# Say "y" when prompted to proceed.

# Activate the environment:
mamba activate myenv
```

Once you've activated, install required packages (they'll be installed in the conda environment):
```bash
pip install -r central/test/requirements.txt
```

## Templating Overview
The 2 main folders you need are `central/` and `templating/`.

### Edit YAML Files to construct fields + note templates
1. Navigate to `central/templating/prompts/ercp/yaml/` folder.
2. Open the base fields file `fields_base.yaml` to see common fields used across all modules.
3. Go to `modules` subfolder. Create a specific module file (e.g. `0.2_stone_extraction.yaml` already exists) to see module-specific fields.
4. Edit or add fields as needed, following the structure outlined in [README_TECH.md](../../README_TECH.md).
5. Save the YAML files after making changes.

## Testing the YAML
### `demo_ercp_pipeline.py`: See note structure with dummy data
This tests: dummy data -> note formatting using the generated templates from the YAML.

```bash
# keep --proc=ercp_base for ercp
# --module is `module_id` listed in the yaml file's metadata section
# --demo_data_json is the demo data file located in `templating/demo_data/`
python templating/demo_ercp_yaml_pipeline.py --proc=ercp_base --module=0.2 --demo_data_json=0.2stone.json5
```
Add as many test json files as needed in the `templating/demo_data/` folder to test different scenarios. Fields in the JSON file should match those defined in the corresponding YAML fields. Fields not present in the JSON will be treated as empty.

The demo pipeline will generate a procedure note using the demo data. The output note will be saved in `templating/demo_ercp_report.txt`.

For example, to generate a note for "ERCP with stone extraction", using the demo data file `0.2stone.json5`, run:
```bash
python templating/demo_ercp_yaml_pipeline.py --proc=ercp_base --module=0.2 --demo_data_json=0.2stone.json5
```
The terminal will display this:
```
Procedure type: stone_extraction
Loading field definitions from: /Users/emilyguan/Downloads/EndoScribe/endo-templating/prompts/ercp/yaml/modules/0.2_stone_extraction.yaml
  Generated prompt: prompts/ercp/subtypes/generated_stone_extraction_prompt.txt
  Generated base template: drafters/procedures/ercp/generated_stone_extraction.yaml
  Generated Pydantic model: models/generated_stone_extraction_model.py

Using demo data file (templating/demo_data/0.2stone.json5)...

Procedure note written to: /Users/emilyguan/Downloads/EndoScribe/endo-templating/templating/demo_ercp_report.txt
```
I can open `templating/demo_ercp_report.txt` to see the generated procedure note. I can also open the generated prompt, base template, and Pydantic model in their respective folders to see how they were created based on the field definitions and demo data.

--------------------------

### `test_subtype.py`: Test LLM extraction as well as note structure with dummy TRANSCRIPT
This pipeline tests: dummy transcript -> LLM extraction -> note formatting using the generated templates from the YAML.

Add test transcripts to `test_subtype.py` with a key (test name) and value (transcript).
```bash
python central/test/test_subflow.py --test=stone_extraction_2 --yaml=0.2_stone_extraction.yaml
# test-name is the key in the test transcript
# yaml is the module yaml file to use
```
The demo pipeline will generate a procedure note saved in `central/test/test_output/`.

For example, to generate a note for "ERCP with stone extraction", using the test transcript with key `stone_extraction_2`, run:
```bash
python central/test/test_subflow.py --test=stone_extraction_2 --yaml=0.2_stone_extraction.yaml
```
The terminal will display this:
```
TEST FOR stone_extraction_2 
============================================================
Loading single template for test: /Users/emilyguan/Downloads/EndoScribe/endoscribe/templating/prompts/ercp/yaml/modules/0.2_stone_extraction.yaml
   Artifacts written to /Users/emilyguan/Downloads/EndoScribe/endoscribe/central/test/test_output
     - Prompt: /Users/emilyguan/Downloads/EndoScribe/endoscribe/central/test/test_output/generated_stone_extraction_prompt.txt
     - Model: /Users/emilyguan/Downloads/EndoScribe/endoscribe/central/test/test_output/generated_stone_extraction_model.py
     - Drafter YAML: /Users/emilyguan/Downloads/EndoScribe/endoscribe/central/test/test_output/generated_stone_extraction_drafter.yaml

Extracting fields...
Wrote LLM-extracted fields to: /Users/emilyguan/Downloads/EndoScribe/endoscribe/central/test/test_output/test/test_subtype_stone_extraction_2.txt

Generating note...
   Generated markdown note

Final test note written to: /Users/emilyguan/Downloads/EndoScribe/endoscribe/central/test/test_output/note_test_subtype_stone_extraction_2.txt
```
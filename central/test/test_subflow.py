import sys
import argparse
from pathlib import Path

# Add project root to path
_here = Path(__file__).parent.parent
_root = _here.parent
sys.path.insert(0, str(_root))

from central.extraction.field_extractor import FieldExtractor
from central.orchestration.orchestrator import Orchestrator

TEST_TRANSCRIPTS = {
    "stone_extraction_1": """
We're here with a 60 year old male for an ERCP for choledocholithiasis. The patient has elevated liver enzymes and bilirubin. The patient recently had a hospital admission for COVID-19 and acute pancreatitis. CT scan show distal common bile duct stone with associated intrahepatic biliary ductal dilation.

Duodenoscope advanced to the second portion of the duodenum without difficulty.
The major papilla looks edematous with evidence of prior sphincterotomy.

Cannulation achieved with guidewire assistance.
Cholangiogram showing multiple filling defects in the common bile duct consistent with stones, largest measuring approximately 12mm.

Biliary sphincterotomy is extended using a sphincterotome. Balloon sweep is being performed, I'm extracting several small stones.
For the large stone, I'm doing endoscopic papillary large balloon dilation using a CRE balloon inflated to 12mm for 60 seconds. Balloon waist was present and resolved with continued inflation.
I successfully extracted the large stone with a Dormia basket. Final sweep showing complete clearance. 

Patient tolerated the procedure well with no immediate complications.
""",

    "stone_extraction_2": """
This is a 65-year-old female patient here for an ERCP procedure for the evaluation and treatment of choledocholithiasis. She's been having right upper quadrant pain and jaundice for around 2 weeks. No history of pancreatitis. She's taking aspirin and metformin. She's in the prone position. Monitored anesthesia care is provided.

A therapeutic duodenoscope was advanced to the second portion of the duodenum.
The major papilla was identified. There was no periampullary diverticulum.

Cannulating of the bile duct using a sphincterotome with guidewire assistance.
Got a cholangiogram which demonstrates multiple filling defects in the distal common bile duct consistent with stones, the largest measuring approximately 12 mm.

A biliary sphincterotomy was performed using electrocautery with good hemostasis.
Initial stone extraction was attempted with a balloon sweep. 
This was partially successful but a large 12mm stone remained.

Due to the large stone, endoscopic papillary large balloon dilation was performed.
A CRE balloon was used with target diameter 12 mm.
Inflation was performed using graded stepwise strategy for 60 seconds.
Balloon waist was present and resolved. Overall outcome was adequate papillary orifice enlargement achieved.

After dilation, balloon sweeps successfully cleared all remaining stones. All visible stones and sludge were removed.
Final balloon occlusion cholangiogram show no filling defects.
"""
}

def test_subflow(transcript: str, output_filename: str, yaml_filename: str = '0.2_stone_extraction.yaml'):
    """ Test a subflow: single module only. 
    Tests FieldExtractor with one module YAML and drafting"""
    # Use orchestrator only to obtain a configured LLM client (or adjust to None to use defaults)
    orchestrator = Orchestrator()
    
    yaml_path = Path(_root) / 'templating' / 'prompts' / 'ercp' / 'yaml' / 'modules' / yaml_filename
    print(f"Loading single template for test: {yaml_path}")
    # Strip any $extends so module does NOT pull in the base template
    import yaml
    with open(yaml_path, 'r', encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    if '$extends' in raw_cfg:
        del raw_cfg['$extends']
    template = raw_cfg

    extractor = FieldExtractor(
        template,
        llm_client=orchestrator.llm_client,
        output_dir=_here / "test" / "test_output"
    )
    print("\nExtracting fields...")
    extracted_data = extractor.extract_fields(transcript)

    # Write extracted data
    out_path = _here / "test" / "test_output" / f"test/{output_filename}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding="utf-8") as f:
        f.write(str(extracted_data.model_dump()))
    print(f"Wrote LLM-extracted fields to: {out_path}\n")

    # Generate note using drafter
    print("Generating note...")
    final_note = orchestrator.generate_note(template, extracted_data)
    note_output_path = _here / "test" / "test_output" / f"note_{output_filename}"
    with open(note_output_path, 'w', encoding="utf-8") as f:
        f.write(final_note)
    print(f"\nFinal test note written to: {note_output_path}\n")

    return final_note


if __name__ == "__main__":
    """ python central/test/test_subflow.py --test=stone_extraction_2 --yaml=0.2_stone_extraction.yaml """

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True, help='Key in TEST_TRANSCRIPTS to run')
    parser.add_argument('--yaml', required=True, help='YAML filename under prompts/ercp/yaml/modules to test')
    args = parser.parse_args()

    try:
        test_name = args.test
        yaml_filename = args.yaml
        if test_name not in TEST_TRANSCRIPTS:
            raise ValueError(f"Unknown test name: {test_name}. Valid keys: {list(TEST_TRANSCRIPTS.keys())}")

        print(f"TEST FOR {test_name} \n"+"="*60)
        output_filename = f"test_subtype_{test_name}.txt"
        result1 = test_subflow(TEST_TRANSCRIPTS[test_name], output_filename, yaml_filename)

    except Exception as e:
        print(f"\n FAIL: Stone extraction test failed: {e}")
        import traceback
        traceback.print_exc()
    
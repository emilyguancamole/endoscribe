import sys
from pathlib import Path

# Add project root to path
_here = Path(__file__).parent.parent
_root = _here.parent
sys.path.insert(0, str(_root))

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

def test_full_flow(transcript: str, output_filename: str = "test_note.txt"):

    orchestrator = Orchestrator()
    result = orchestrator.process_transcript(transcript) # full pipeline
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    classification = result['classification']
    print(f"Base Template: {classification.base_template}")
    print(f"\nKey Findings:")

    print("\n" + "=" * 80)
    print("ASSEMBLED TEMPLATE")
    merged = result['merged_template']
    print(f"Procedure Group: {merged['meta']['procedure_group']}")
    print(f"Active Modules: {merged['meta'].get('active_modules', [])}")
    # Write generated note
    output_path = _here / f"test/{output_filename}"
    with open(output_path, 'w') as f:
        f.write(result['final_note'])
    print(f"\nRendered test procedure note written to: {output_path}\n")
    
    return result


if __name__ == "__main__":
    """ python central/test/test_full_flow.py """
    try:
        test_name = "stone_extraction_2"
        
        print(f"TEST FOR {test_name} \n"+"="*60)
        output_filename = f"test_full_{test_name}.txt"
        result1 = test_full_flow(TEST_TRANSCRIPTS[test_name], output_filename)
    except Exception as e:
        print(f"\n FAIL: Stone extraction test failed: {e}")
        import traceback
        traceback.print_exc()
    
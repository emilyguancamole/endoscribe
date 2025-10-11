import argparse
import os
import pandas as pd
from drafters import ColonoscopyDrafter, EUSDrafter, ERCPDrafter, EGDDrafter

def main():
    '''
    Will overwrite any existing reports with same name.
    
    Example use:
        python drafter.py --procedure=col --pred_csv=results/colonoscopy/llama_col_outputs.csv --polyp_csv=results/colonoscopy/llama_polyp_outputs.csv --output_dir=/Users/emilyguan/Downloads/EndoScribe/reports_ai/colonoscopy/abstract/llama_extracted --samples_to_process 16 115 95

        python drafter.py --procedure=eus --pred_csv=results/eus/llama_outputs.csv --output_dir=/Users/emilyguan/Downloads/EndoScribe/reports_ai/eus --samples_to_process mass02 cancer07

        python drafter.py --procedure=ercp --pred_csv=results/ercp/long-10-2025.csv --output_dir=drafters/results/ercp/longform --samples_to_process all
    
        python drafter.py --procedure=egd --pred_csv=results/egd/082025-test.csv --output_dir=drafters/results/egd/longform --samples_to_process egd01 egd02 egd03
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--procedure', type=str, required=True, choices=['col', 'eus', 'ercp', 'egd'])
    parser.add_argument('--pred_csv', required=True, help="Path to predictions csv") #? should this be from db directly instead??
    parser.add_argument('--polyp_csv', required=False, help="Path to corresponding polyp predictions csv, for colonoscopy only")
    parser.add_argument('--patients_data', default="data/patients.csv", help="Path to patient info data from RedCap") 
    parser.add_argument('--procedures_data', default="data/procedures.csv", help="Path to patient info data from RedCap") 
    parser.add_argument('--output_dir', required=True, help="Directory to save reports")
    parser.add_argument('--transcripts', default='transcription/eus_transcripts/initial_whisper_lg_v3.csv', help="File with AI transcripts so we can add to report for reference")
    parser.add_argument('--samples_to_process', nargs='*', help="List of sample numbers to process")
    args = parser.parse_args()  
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Make first column (procedure id) the index
    pred_df = pd.read_csv(args.pred_csv, index_col=0, dtype={0: str})

    # Require polyp_csv for colonoscopy
    if args.procedure == "col":
        if not args.polyp_csv:
            raise ValueError("For colonoscopy reports, provide a path to the polyp predictions, --polyp_csv")
        polyp_df = pd.read_csv(args.polyp_csv, index_col=0, dtype={0: str})

    if args.samples_to_process[0] == "all":
        samples_to_process = pred_df.index.tolist()
    else:
        samples_to_process = args.samples_to_process

    # Create drafter class based on procedure
    drafter_classes = {
        "col": ColonoscopyDrafter,
        "eus": EUSDrafter,
        "ercp": ERCPDrafter,
        "egd": EGDDrafter,
    }
    Drafter = drafter_classes[args.procedure]
    for sample in samples_to_process:
        if args.procedure == "col":
            drafter = Drafter(sample, pred_df, polyp_df, patients)
        else:
            drafter = Drafter(sample, pred_df)

        doc = drafter.draft_doc()
        doc.save(f"{args.output_dir}/{sample}.docx")
        print(f"Report for '{sample}' created at {args.output_dir}/{sample}.docx")

if __name__ == "__main__":
    main()
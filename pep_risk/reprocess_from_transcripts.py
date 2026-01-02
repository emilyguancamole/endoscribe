import os
import sys
import argparse
import pandas as pd
import asyncio
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import LLMClient
from processors.ercp_processor import ERCPProcessor
from pep_risk.peprisc_model import predict_pep_risk
from pep_risk.server import load_manual_data_from_ground_truth, save_batch_prediction_results
from pep_risk.evaluation import evaluate_against_ground_truth

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "pep_risk" / "results"


async def process_single_file(filename: str, transcript: str, processor: ERCPProcessor, 
                               eval_csv_path: str) -> dict:
    """
    Process a single file using its existing transcript:
    - Re-run LLM extraction
    - Load manual data from ground truth
    - Run R model prediction
    - Save results
    """
    print(f"\n{'='*80}")
    print(f"Processing: {filename}")
    print(f"{'='*80}")
    
    print(f"Re-extracting PEP risk factors...")
    try:
        extraction = processor.extract_pep_from_transcript(transcript, filename)
        print(f" LLM extraction complete: {len(extraction)} fields")
    except Exception as e:
        print(f" LLM extraction error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "step": "llm_extraction", "filename": filename}
    
    print(f"Loading manual patient data from ground truth...")
    manual_data = load_manual_data_from_ground_truth(filename)
    if manual_data is None:
        print(f"Warning: No manual data found for {filename}, using defaults")
        manual_data = {}
    
    print(f"Running R PEPRISC model prediction...")
    try:
        prediction_result = predict_pep_risk(
            manual_data=manual_data,
            llm_extracted_data=extraction
        )
        if prediction_result.get("success"):
            treatment_count = len(prediction_result.get("treatment_predictions", []))
            baseline_risk = prediction_result.get("risk_score")
            print(f" R model prediction:")
            print(f"  - Baseline risk: {baseline_risk}% ({prediction_result.get('risk_category')})")
            print(f"  - Treatment scenarios: {treatment_count}")
    except Exception as e:
        print(f" R model error: {e}")
        import traceback
        traceback.print_exc()
        prediction_result = {"success": False, "error": str(e)}
    
    print(f"Saving results to CSV...")
    try:
        save_batch_prediction_results(
            filename=filename,
            manual_data=manual_data,
            llm_extraction=extraction,
            prediction_result=prediction_result,
            transcript=transcript
        )
        print(f" Results saved to batch_pep_predictions.csv")
    except Exception as e:
        print(f" Failed to save batch results: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        ground_truth_csv = os.path.join(BASE_DIR, "pep_risk", "GROUND_TRUTH.csv")
        evaluate_against_ground_truth(
            original_filename=filename,
            extraction=extraction,
            ground_truth_csv=ground_truth_csv,
            save_to_csv=eval_csv_path
        )
        print(f" Evaluation saved to pep_eval.csv")
    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
    
    return {
        "filename": filename,
        "success": prediction_result.get("success", False),
        "baseline_risk": prediction_result.get("risk_score"),
        "extraction_fields": len(extraction)
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Re-process files using existing transcriptions"
    )
    parser.add_argument(
        "--input",
        default="pep_risk/results/batch_pep_predictions.csv",
        help="Path to CSV with existing transcriptions"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific filenames to re-process (optional, otherwise processes all)"
    )
    parser.add_argument(
        "--output-dir",
        default="pep_risk/results",
        help="Output directory for results"
    )
    args = parser.parse_args()
    
    input_csv = Path(args.input)
    if not input_csv.exists():
        print(f"Error: Input CSV not found: {input_csv}")
        sys.exit(1)
    
    print(f"Loading transcriptions from {input_csv}...")
    df = pd.read_csv(input_csv)
    if "transcript" not in df.columns or "filename" not in df.columns:
        print("Error: CSV must have 'filename' and 'transcript' columns")
        sys.exit(1)
    
    if args.files:
        df = df[df['filename'].isin(args.files)]
        if len(df) == 0:
            print(f"Error: No matching files found for: {args.files}")
            sys.exit(1)
        print(f"Processing {len(df)} specified files")
    else:
        print(f"Processing all {len(df)} files from CSV")
    
    df = df[df['transcript'].notna() & (df['transcript'] != "")]
    
    # Initialize processor
    try:
        llm_handler = LLMClient.from_config("openai_gpt4o")
    except Exception as e:
        print("Failed to initialize LLM from config 'openai_gpt4o':", e)
        llm_handler = None
    print("Initializing ERCP processor...")
    processor = None
    if llm_handler is not None:
        try:
            processor = ERCPProcessor(
                procedure_type="ercp",
                system_prompt_fp=os.path.join(BASE_DIR, "pep_risk", "prompts", "system.txt"),
                output_fp=os.path.join(BASE_DIR, "pep_risk", "results", "pep_server.csv"),
                llm_handler=llm_handler,
                to_postgres=False,
            )
        except Exception as e:
            print("Failed to initialize ERCPProcessor:", e)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_csv_path = str(output_dir / "pep_eval.csv")
    
    # Process each file
    results = []
    for _, row in df.iterrows():
        filename = row['filename']
        transcript = row['transcript']
        
        try:
            result = await process_single_file(
                filename=filename,
                transcript=transcript,
                processor=processor,
                eval_csv_path=eval_csv_path
            )
            results.append(result)
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "filename": filename,
                "success": False,
                "error": str(e)
            })
    print(f"\nResults saved to:")
    print(f"  - Batch predictions: {output_dir / 'batch_pep_predictions.csv'}")
    print(f"  - Evaluations: {output_dir / 'pep_eval.csv'}")


if __name__ == "__main__":
    """
    Re-run PEP LLM extraction and R model predictions using existing transcriptions.
    Reads transcriptions from batch_pep_predictions.csv and:
    - Re-runs LLM extraction with updated prompts
    - Loads manual data from ground truth
    - Re-runs R model predictions
    - Saves updated results to CSV

    Usage:
    Re-process all files:
        python3 pep_risk/reprocess_from_transcripts.py
    Re-process specific files:
    python3 pep_risk/reprocess_from_transcripts.py --files 1.mp3 11862-4.mp3 11862-5.mp3
    Use a different input CSV, Specify output directory:
        python pep_risk/reprocess_from_transcripts.py --input pep_risk/results/batch_pep_predictions.csv --output-dir ./results_v2
    """
    asyncio.run(main())

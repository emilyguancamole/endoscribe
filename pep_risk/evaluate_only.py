#!/usr/bin/env python3
"""
Standalone evaluation script for comparing ERCP extractions to ground truth.
No transcription or LLM calls - just evaluation.

Usage:
    python pep_risk/evaluate_only.py <session_id>
    python pep_risk/evaluate_only.py 1234 --extraction-json path/to/extraction.json
    python pep_risk/evaluate_only.py 1234 --out-json results/eval.json
"""
import argparse
import json
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from pep_risk.evaluation import (
    evaluate_against_ground_truth,
    map_ground_truth_columns,
    compare_extraction_to_truth
)


def load_extraction_from_sessions(session_id: str, sessions_csv: str):
    """Load extraction dict from the sessions CSV by session_id."""
    if not os.path.exists(sessions_csv):
        raise FileNotFoundError(f"Sessions CSV not found: {sessions_csv}")
    
    df = pd.read_csv(sessions_csv, dtype=str).fillna("")
    
    # Try to find the session by session_id column
    if "session_id" in df.columns:
        row = df[df["session_id"] == session_id]
    else:
        # Fallback: search all columns for the session_id
        row = df[df.apply(lambda r: session_id in str(r.values), axis=1)]
    
    if row.empty:
        raise ValueError(f"Session '{session_id}' not found in {sessions_csv}")
    
    row = row.iloc[0]
    
    # Try to find extraction as a JSON column
    for col in ["extraction", "extractions", "llm_extraction", "ercp_extraction"]:
        if col in row.index and row[col]:
            try:
                return json.loads(row[col])
            except Exception:
                # Maybe stored as dict string; attempt eval
                try:
                    return eval(row[col])
                except Exception:
                    pass
    
    # If extraction fields are flattened as columns, collect them
    possible_fields = [
        "indications", "samples_taken", "egd_findings", "ercp_findings",
        "biliary_stent_type", "pd_stent", "impressions",
        "pancreatic_sphincterotomy", "precut_sphincterotomy",
        "minor_papilla_sphincterotomy", "failed_cannulation",
        "difficult_cannulation", "acinarization",
        "guidewire_cannulation", "biliary_sphincterotomy",
        "indomethacin_nsaid_prophylaxis", "aggressive_hydration",
        "pancreatic_duct_stent_placement",
        "pneumatic_dilation_of_intact_biliary_sphincter",
        "pancreatic_duct_injections", "pancreatic_duct_injections_2",
        "pancreo_biliary_malignancy",
        "guidewire_passage_into_pancreatic_duct",
        "guidewire_passage_into_pancreatic_duct_2"
    ]
    
    extraction = {k: v for k, v in row.items() if k in possible_fields}
    if extraction:
        return extraction
    
    raise ValueError("Could not find extraction data in sessions CSV row.")


def find_gt_row_for_session(session_id: str, ground_truth_csv: str):
    """Find the ground truth row matching this session_id."""
    if not os.path.exists(ground_truth_csv):
        raise FileNotFoundError(f"Ground truth CSV not found: {ground_truth_csv}")
    
    gt = pd.read_csv(ground_truth_csv, dtype=str).fillna("")
    
    # Try exact match on record_id
    candidates = gt[gt["record_id"] == session_id]
    if candidates.empty:
        # Try with common extensions
        for ext in [".mp3", ".wav", ".m4a"]:
            candidates = gt[gt["audio_recording"] == f"{session_id}{ext}"]
            if not candidates.empty:
                break
    if candidates.empty:
        # Try substring match
        candidates = gt[gt["record_id"].str.contains(session_id, na=False)]
    
    if candidates.empty:
        raise ValueError(f"No ground truth row found matching record_id for session '{session_id}'")
    
    return candidates.iloc[0]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extraction vs ground truth (no re-transcription).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using session from ercp_sessions.csv
  python pep_risk/evaluate_only.py 1234

  # Evaluate using explicit extraction JSON
  python pep_risk/evaluate_only.py 1234 --extraction-json path/to/extraction.json

  # Save output to JSON
  python pep_risk/evaluate_only.py 1234 --out-json results/eval.json
        """
    )
    parser.add_argument(
        "session_id",
        help="Session ID (for prerecorded files, use filename stem like '1234')"
    )
    parser.add_argument(
        "--sessions-csv",
        default="pep_risk/results/pep_llm_extraction.csv",
        help="Path to sessions CSV (default: pep_risk/results_longform/pep_llm_extraction.csv)"
    )
    parser.add_argument(
        "--ground-truth",
        default="pep_risk/ground_truth.csv",
        help="Path to ground truth CSV (default: pep_risk/ground_truth.csv)"
    )
    parser.add_argument(
        "--extraction-json",
        help="Path to JSON file with extraction dict (skip reading sessions CSV)"
    )
    parser.add_argument(
        "--out-json",
        help="Optional path to save evaluation result JSON"
    )
    
    args = parser.parse_args()
    
    # Load extraction
    try:
        if args.extraction_json:
            print(f"Loading extraction from {args.extraction_json}")
            with open(args.extraction_json, "r") as f:
                extraction = json.load(f)
        else:
            print(f"Loading extraction for session '{args.session_id}' from {args.sessions_csv}:")
            extraction = load_extraction_from_sessions(args.session_id, args.sessions_csv)
            print(f"Loaded extraction: {extraction}")
    except Exception as e:
        print(f"Error loading extraction: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Find ground truth row
    try:
        print(f"Finding ground truth row for session '{args.session_id}'")
        gt_row = find_gt_row_for_session(args.session_id, args.ground_truth)
        print(f"Found ground truth: audio_recording={gt_row}")
    except Exception as e:
        print(f"Error loading ground truth: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Map and evaluate
    truth_mapped = map_ground_truth_columns(gt_row)
    eval_result = compare_extraction_to_truth(extraction, truth_mapped)
    
    # Prepare output
    output = {
        "session_id": args.session_id,
        "evaluation": eval_result,
        "extraction_sample": {k: extraction.get(k) for k in list(truth_mapped.keys())[:10]},
        "truth_sample": {k: v for k, v in list(truth_mapped.items())[:10]},
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS FOR SESSION: {args.session_id}")
    print("="*60)
    print(f"\nOverall Accuracy: {eval_result['overall_accuracy']:.1%}")
    print(f"\nPer-field results:")
    for field, result in eval_result.get("per_field", {}).items():
        if isinstance(result, dict) and "match" in result:
            status = "✓" if result["match"] else "✗"
            print(f"  {status} {field}: extracted={result.get('extracted')} vs truth={result.get('truth')}")
    
    print("\n" + "="*60)
    
    # Save to JSON if requested
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved evaluation to {args.out_json}")


if __name__ == "__main__":
    main()

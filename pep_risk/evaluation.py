"""
Shared evaluation utilities for comparing ERCP extraction results to ground truth.
Used by both the server and standalone evaluation scripts.
"""
import math
import os
from typing import Dict, List, Optional
import pandas as pd


def safe_float(val):
    """Convert value to float, returning 0.0 for NaN/inf/None."""
    try:
        if val is None:
            return 0.0
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0


def _unwrap_scalar(val):
    """
    Convert numpy/pandas scalar types to native Python types when possible.
    If the object exposes `.item()` (like numpy scalars or pandas NA types), use it.
    """
    try:
        # pandas/Numpy scalar
        if hasattr(val, "item") and callable(val.item):
            return val.item()
    except Exception:
        pass
    return val


def normalize_text(s: Optional[str]) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def parse_list_field(val: Optional[object]) -> List[str]:
    """Parse list-like fields from either list or string with separators."""
    if val is None:
        return []
    if isinstance(val, list):
        return [normalize_text(x) for x in val if str(x).strip()]
    # split on common delimiters
    parts = [p.strip() for p in str(val).replace(";", ",").split(",")]
    return [normalize_text(p) for p in parts if p]


def map_ground_truth_columns(row: pd.Series) -> Dict[str, object]:
    """
    Map/rename ground-truth columns to match extraction schema.
    Args:
        row: A pandas Series representing one ground truth row
    Returns:
        Dictionary mapping extraction field names to ground truth values
    """
    column_mapping = {
        # ground_truth_column: extraction_field_name
        "malignancy": "pancreo_biliary_malignancy",
        "guidewire_passes": "guidewire_passage_into_pancreatic_duct",
        "guidewire_passes_number": "guidewire_passage_into_pancreatic_duct_2",
        "pd_injections": "pancreatic_duct_injections",
        "pd_injections_number": "pancreatic_duct_injections_2",
        "papilla_sphincterotomy": "minor_papilla_sphincterotomy",
        "pnuematic_dilation": "pneumatic_dilation_of_intact_biliary_sphincter",
    }

    mapped = {}
    # Apply explicit remaps
    for gt_col, ext_col in column_mapping.items():
        if gt_col in row:
            mapped[ext_col] = _unwrap_scalar(row[gt_col])
    
    # For columns that already match extraction field names, copy them directly
    standard_fields = [
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
    
    for col in standard_fields:
        if col in row.index and col not in mapped:
            mapped[col] = _unwrap_scalar(row[col])
    
    return mapped


def normalize_boolean_value(val):
    """
    Normalize a value to a standard boolean representation.
    Treats False, 0, None, empty string, and "nan" as False.
    Treats True, 1, and non-empty strings as True.
    """
    # Unwrap numpy/pandas scalars first
    val = _unwrap_scalar(val)
    if val is None or val == "" or str(val).lower() in ["nan", "none", ""]:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val) and val != 0
    # String values
    str_val = str(val).strip().lower()
    if str_val in ["false", "0", "", "no", "n", "none", "nan"]:
        return False
    return True


def compare_extraction_to_truth(extraction: Dict[str, object], truth: Dict[str, object]) -> Dict:
    """
    Compare extraction dict to truth dict field-by-field.
    
    Returns metrics dictionary with:
        - overall_accuracy: float between 0 and 1
        - per_field: dict of field-level results
        
    Note: For boolean/numeric fields, False/0/None/""/NaN are all treated as equivalent (falsy),
    and True/1 are treated as equivalent (truthy).
    """
    fields = [
        "indications",
        "samples_taken",
        "egd_findings",
        "ercp_findings",
        "biliary_stent_type",
        "pd_stent",
        "impressions",
        "pancreatic_sphincterotomy",
        "precut_sphincterotomy",
        "minor_papilla_sphincterotomy",
        "failed_cannulation",
        "difficult_cannulation",
        "pneumatic_dilation_of_intact_biliary_sphincter",
        "pancreatic_duct_injections",
        "pancreatic_duct_injections_2",
        "acinarization",
        "pancreo_biliary_malignancy",
        "guidewire_cannulation",
        "guidewire_passage_into_pancreatic_duct",
        "guidewire_passage_into_pancreatic_duct_2",
        "biliary_sphincterotomy",
        "indomethacin_nsaid_prophylaxis",
        "aggressive_hydration",
        "pancreatic_duct_stent_placement"
    ]

    results = {}
    matches = []
    
    for f in fields:
        ext_val = _unwrap_scalar(extraction.get(f))
        gt_val = _unwrap_scalar(truth.get(f))
        
        # Special handling for list fields (like impressions)
        if f == "impressions":
            ext_list = set(parse_list_field(ext_val))
            gt_list = set(parse_list_field(gt_val))
            tp = len(ext_list & gt_list)
            fp = len(ext_list - gt_list)
            fn = len(gt_list - ext_list)
            precision = tp / (tp + fp) if (tp + fp) else 1.0
            recall = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 1.0
            results[f] = {
                "precision": float(safe_float(precision)),
                "recall": float(safe_float(recall)),
                "f1": float(safe_float(f1)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
            }
            # Consider a match when F1 is perfect
            matches.append(1 if f1 == 1.0 else 0)
        # Boolean/numeric fields - normalize to boolean for comparison
        elif f in [
            "pancreatic_sphincterotomy", "precut_sphincterotomy",
            "minor_papilla_sphincterotomy", "failed_cannulation",
            "difficult_cannulation", "acinarization",
            "guidewire_cannulation", "biliary_sphincterotomy",
            "indomethacin_nsaid_prophylaxis", "aggressive_hydration",
            "pancreatic_duct_stent_placement", "samples_taken", "pd_stent",
            "pneumatic_dilation_of_intact_biliary_sphincter",
            "pancreo_biliary_malignancy"
        ]:
            ext_bool = bool(normalize_boolean_value(ext_val))
            gt_bool = bool(normalize_boolean_value(gt_val))
            match = bool(ext_bool == gt_bool)
            results[f] = {"match": bool(match), "extracted": bool(ext_bool), "truth": bool(gt_bool)}
            matches.append(1 if match else 0)
        # Numeric count fields - compare as numbers but treat empty/None as 0
        elif f in ["pancreatic_duct_injections_2", "guidewire_passage_into_pancreatic_duct_2"]:
            try:
                ext_num = float(safe_float(ext_val)) if ext_val not in [None, "", "nan", "NaN"] else 0.0
                gt_num = float(safe_float(gt_val)) if gt_val not in [None, "", "nan", "NaN"] else 0.0
                match = bool(abs(ext_num - gt_num) < 1e-6)
                results[f] = {"match": bool(match), "extracted": float(ext_num), "truth": float(gt_num)}
                matches.append(1 if match else 0)
            except Exception:
                # Fallback to boolean comparison
                ext_bool = normalize_boolean_value(ext_val)
                gt_bool = normalize_boolean_value(gt_val)
                match = (ext_bool == gt_bool)
                results[f] = {"match": match, "extracted": ext_bool, "truth": gt_bool}
                matches.append(1 if match else 0)
        # Text fields with boolean semantics
        elif f in ["pancreatic_duct_injections", "guidewire_passage_into_pancreatic_duct"]:
            # These might be stored as True/False or 0/1 in extraction but need boolean comparison
            ext_bool = bool(normalize_boolean_value(ext_val))
            gt_bool = bool(normalize_boolean_value(gt_val))
            match = bool(ext_bool == gt_bool)
            results[f] = {"match": bool(match), "extracted": bool(ext_bool), "truth": bool(gt_bool)}
            matches.append(1 if match else 0)
        # Text fields (indications, findings, etc)
        else:
            # For text fields, None should match None, and empty should match empty
            ext_norm = normalize_text(ext_val) if ext_val not in [None, "", "nan", "NaN"] else ""
            gt_norm = normalize_text(gt_val) if gt_val not in [None, "", "nan", "NaN"] else ""
            match = bool(ext_norm == gt_norm)
            # Ensure we store native python types for extracted/truth
            results[f] = {"match": bool(match), "extracted": _unwrap_scalar(ext_val), "truth": _unwrap_scalar(gt_val)}
            matches.append(1 if match else 0)

    overall = safe_float(sum(matches) / len(matches)) if matches else 0.0
    return {"overall_accuracy": float(round(overall, 3)), "per_field": results}


def evaluate_against_ground_truth(
    original_filename: str,
    extraction: dict,
    ground_truth_csv: str,
    save_to_csv: Optional[str] = None
) -> dict:
    """
    Evaluate extraction by matching ground truth row by audio_recording column.
    
    Args:
        original_filename: The audio filename to match in ground truth (e.g., "1234.mp3")
        extraction: Dictionary of extracted fields
        ground_truth_csv: Path to ground truth CSV file
        save_to_csv: Optional path to save evaluation results CSV
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not os.path.exists(ground_truth_csv):
        return {"warning": f"Ground truth file not found at {ground_truth_csv}"}

    try:
        gt_df = pd.read_csv(ground_truth_csv)
    except Exception as e:
        return {"warning": f"Failed to read ground truth CSV: {e}"}

    # Expect a column named 'audio_recording' with filenames like '1234.mp3'
    if "record_id" not in gt_df.columns:
        return {"warning": "Missing 'record_id' column in ground truth CSV"}

    row = gt_df[gt_df["record_id"].apply(lambda x: os.path.splitext(str(x))[0]) == os.path.splitext(original_filename)[0]]
    if row.empty:
        return {"warning": f"No ground truth row for record_id={original_filename}"}

    row = row.iloc[0]
    truth_mapped = map_ground_truth_columns(row)
    metrics = compare_extraction_to_truth(extraction, truth_mapped)
    
    # Save evaluation row if requested
    if save_to_csv:
        eval_row = {
            "audio_recording": original_filename,
            "session_id": os.path.splitext(original_filename)[0],
            "overall_accuracy": metrics.get("overall_accuracy", 0.0),
        }
        
        # Flatten per_field results into columns - store match/extracted/truth for each field
        per_field = metrics.get("per_field", {})
        for field_name, field_result in per_field.items():
            # For most fields, format as "match/extracted/truth"
            if isinstance(field_result, dict) and "match" in field_result:
                match = bool(field_result["match"])
                extracted = field_result.get("extracted", "")
                truth = field_result.get("truth", "")
                # Convert to string representation for CSV
                eval_row[f"field_{field_name}"] = f"{match}/{extracted}/{truth}"
            # For list fields (like impressions), use F1 metrics
            elif isinstance(field_result, dict) and "f1" in field_result:
                match = bool(field_result["f1"] == 1.0)
                f1 = field_result.get("f1", 0.0)
                precision = field_result.get("precision", 0.0)
                recall = field_result.get("recall", 0.0)
                eval_row[f"field_{field_name}"] = f"{match}/F1={f1:.2f}/P={precision:.2f},R={recall:.2f}"
            else:
                # Fallback: store the result as-is
                eval_row[f"field_{field_name}"] = str(field_result)
        
        if os.path.exists(save_to_csv):
            edf = pd.read_csv(save_to_csv)
            edf = pd.concat([edf, pd.DataFrame([eval_row])], ignore_index=True)
        else:
            edf = pd.DataFrame([eval_row])
        edf.to_csv(save_to_csv, index=False)
        print(f"Saved EVALUATION results to {save_to_csv}")

    return metrics

"""Run R PEPRISC model on `GROUND_TRUTH.csv` and save summarized treatment predictions.

Produces CSV with columns:
  - record_id
  - risk_aggressive_hydration_only_pct
  - risk_indomethacin_only_pct
  - risk_pd_stent_only_pct
  - risk_aggressive_hydration_and_indomethacin_pct
  - risk_indomethacin_and_pd_stent_pct
  - risk_no_treatment_pct
  - baseline_risk_pct

This script uses the existing Python bridge in `pep_risk.peprisc_model` which
invokes the R model via rpy2. Run from repository root with the appropriate
R environment available (rpy2 + R + required R packages).
"""

import os
import csv
import traceback
import pandas as pd

from pep_risk.peprisc_model import predict_pep_risk


GROUND_TRUTH = os.path.join(os.path.dirname(__file__), "results", "GROUND_TRUTH_LONG.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "results", "long_pep_pred_from_ground_truth.csv")


def map_row_to_manual(row: pd.Series) -> dict:
    """Map a GROUND_TRUTH row to the manual_data keys expected by peprisc_model.

    The mapping uses best-effort conversions based on column names in
    `pep_risk/GROUND_TRUTH.csv`.
    """
    def to_bool(v):
        if pd.isna(v):
            return False
        try:
            return int(v) == 1
        except Exception:
            s = str(v).strip().lower()
            return s in ("1", "true", "t", "yes", "y")

    manual = {}
    # Basic demographics
    manual["age_years"] = row.get("age") if "age" in row.index else row.get("age_years")
    # In this dataset `sex` appears coded 1/2 -> treat 1 as male
    if "sex" in row.index:
        try:
            manual["gender_male"] = int(row.get("sex")) == 1
        except Exception:
            manual["gender_male"] = str(row.get("sex")).strip().lower() in ("m", "male", "1")
    else:
        manual["gender_male"] = bool(row.get("gender_male", False))

    manual["bmi"] = row.get("bmi")

    # Direct mappings where available
    manual["cholecystectomy"] = to_bool(row.get("cholecystectomy"))
    manual["history_of_pep"] = to_bool(row.get("pep_history") or row.get("history_of_pep") or row.get("pep"))
    manual["hx_of_recurrent_pancreatitis"] = to_bool(row.get("history_of_pancreatitis"))
    manual["sod"] = to_bool(row.get("sod"))
    manual["trainee_involvement"] = to_bool(row.get("trainee_involvement"))

    # Procedure-level features
    manual["guidewire_cannulation"] = to_bool(row.get("guidewire_cannulation"))
    manual["guidewire_passage_into_pancreatic_duct"] = to_bool(row.get("guidewire_passes"))
    manual["guidewire_passage_into_pancreatic_duct_2"] = row.get("guidewire_passes_number") or 0

    # Sphincterotomy / cannulation features
    manual["biliary_sphincterotomy"] = to_bool(row.get("biliary_sphincterotomy"))
    manual["pancreatic_sphincterotomy"] = to_bool(row.get("pancreatic_sphincterotomy"))
    manual["precut_sphincterotomy"] = to_bool(row.get("precut_sphincterotomy"))
    manual["minor_papilla_sphincterotomy"] = to_bool(row.get("minor_papilla_sphincterotomy"))
    manual["failed_cannulation"] = to_bool(row.get("failed_cannulation"))
    manual["difficult_cannulation"] = to_bool(row.get("difficult_cannulation") or row.get("difficulty_of_ercp"))
    # pneumatic_dilation naming difference
    manual["pneumatic_dilation_of_intact_biliary_sphincter"] = to_bool(row.get("pnuematic_dilation") or row.get("pneumatic_dilation"))

    # Pancreatic duct injections
    manual["pancreatic_duct_injections"] = to_bool(row.get("pd_injections") or row.get("pancreatic_duct_injection"))
    manual["pancreatic_duct_injections_2"] = row.get("pd_injections_number") or row.get("pancreatic_duct_injections_2") or 0

    manual["acinarization"] = to_bool(row.get("acinarization"))
    manual["pancreo_biliary_malignancy"] = to_bool(row.get("malignancy") or row.get("pancreo_biliary_malignancy"))

    # Fallback defaults will be handled by the model prepare function
    return manual


def extract_predictions_to_row(record_id, result):
    # default NaNs
    cols = {
        "risk_aggressive_hydration_only_pct": None,
        "risk_indomethacin_only_pct": None,
        "risk_pd_stent_only_pct": None,
        "risk_aggressive_hydration_and_indomethacin_pct": None,
        "risk_indomethacin_and_pd_stent_pct": None,
        "risk_no_treatment_pct": None,
        "baseline_risk_pct": None,
    }

    if not result or not result.get("success"):
        return {"record_id": record_id, **cols}

    # treatment_predictions: list of {therapy, risk_percentage, risk_category}
    for tp in result.get("treatment_predictions", []):
        therapy = tp.get("therapy", "").strip()
        val = tp.get("risk_percentage")
        if therapy == "Aggressive hydration only":
            cols["risk_aggressive_hydration_only_pct"] = val
        elif therapy == "Indomethacin only":
            cols["risk_indomethacin_only_pct"] = val
        elif therapy == "PD stent only":
            cols["risk_pd_stent_only_pct"] = val
        elif therapy == "Aggressive hydration and indomethacin":
            cols["risk_aggressive_hydration_and_indomethacin_pct"] = val
        elif therapy == "Indomethacin and PD stent":
            cols["risk_indomethacin_and_pd_stent_pct"] = val
        elif therapy == "No treatment":
            cols["risk_no_treatment_pct"] = val

    # baseline_risk_pct: use the No treatment prediction if present, else use final_risk
    cols["baseline_risk_pct"] = cols["risk_no_treatment_pct"] if cols["risk_no_treatment_pct"] is not None else result.get("risk_score")

    return {"record_id": record_id, **cols}


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

    df = pd.read_csv(GROUND_TRUTH, dtype=str, encoding="latin1")

    out_rows = []

    total = len(df)
    print(f"Found {total} records in {GROUND_TRUTH}")

    for idx, row in df.iterrows():
        record_id = row.get("record_id") or f"row_{idx}"
        print(f"Processing {record_id} ({idx+1}/{total})...")

        manual = map_row_to_manual(row)

        try:
            result = predict_pep_risk(manual_data=manual, llm_extracted_data=None)
        except Exception as e:
            print(f"Error predicting for {record_id}: {e}")
            traceback.print_exc()
            result = {"success": False}

        out = extract_predictions_to_row(record_id, result)
        out_rows.append(out)

    # Save CSV with ordered columns
    cols = [
        "record_id",
        "risk_aggressive_hydration_only_pct",
        "risk_indomethacin_only_pct",
        "risk_pd_stent_only_pct",
        "risk_aggressive_hydration_and_indomethacin_pct",
        "risk_indomethacin_and_pd_stent_pct",
        "risk_no_treatment_pct",
        "baseline_risk_pct",
    ]

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote predictions to: {OUT_CSV}")


if __name__ == "__main__":
    main()

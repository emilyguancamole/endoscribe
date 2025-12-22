"""
PEP risk prediction using R-based PEPRISC model.
This module integrates the validated R PEPRISC model for clinical use.
"""

import os
import sys
import pandas as pd
from typing import Dict, Optional
from pep_risk.pep_types import therapy_id_from_label, therapy_label_from_id

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration for R model
R_BRIDGE_PATH = os.getenv("R_BRIDGE_PATH", 
    os.path.expanduser("pep_risk/prediction_model/pep_risk-master/pep_risk_app/bridge_exports.R"))

RISK_THRESHOLDS = {
    "low": 5.0,     
    "moderate": 15.0
}

_R_MODEL = None
def get_r_model():
    """Load and return the R PEPRISC model"""
    global _R_MODEL
    if _R_MODEL is None:
        try:
            from pep_risk.peprisc_bridge import PepriscBridge
            
            bridge_path = R_BRIDGE_PATH
            if not os.path.exists(bridge_path):
                raise FileNotFoundError(f"R bridge file not found at {R_BRIDGE_PATH}. Set R_BRIDGE_PATH environment variable.")
            
            _R_MODEL = PepriscBridge(bridge_path, 'peprisk_predict')
            print(f"✓ R PEPRISC model loaded successfully from: {bridge_path}")
        except Exception as e:
            print(f"✗ Failed to load R PEPRISC model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"R model is required but failed to load: {e}")
    return _R_MODEL


def prepare_r_input_dataframe(combined_data: Dict) -> pd.DataFrame:
    """
    Convert combined risk factors dict to DataFrame format expected by R model.
    
    Args:
        combined_data: Dict with all risk factors (manual + LLM extracted)
    
    Returns:
        pandas DataFrame with one row containing all risk factors
    """
    
    # Helper functions to safely coerce values to the types expected by R
    def _safe_bool_int(val, default=0):
        """Return 1/0 for truthy/falsy values; handle None and strings safely."""
        if val is None:
            return int(bool(default))
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, float)):
            try:
                return int(val)
            except Exception:
                return int(bool(default))
        s = str(val).strip().lower()
        if s in ["", "nan", "none"]:
            return int(bool(default))
        if s in ["1", "true", "t", "y", "yes"]:
            return 1
        if s in ["0", "false", "f", "n", "no"]:
            return 0
        try:
            return int(float(s))
        except Exception:
            return int(bool(default))

    def _safe_int(val, default=0):
        if val is None:
            return int(default)
        try:
            return int(val)
        except Exception:
            try:
                return int(float(str(val)))
            except Exception:
                return int(default)

    def _safe_float(val, default=0.0):
        if val is None:
            return float(default)
        try:
            return float(val)
        except Exception:
            try:
                return float(str(val))
            except Exception:
                return float(default)

    # Map Python field names to R model expectations
    # Note: R model expects integers (0/1) for booleans, not True/False
    df = pd.DataFrame({
        "patient_id": [1],
        "age_years": [_safe_int(combined_data.get("age_years", 50), 50)],
        "gender_male_1": [_safe_bool_int(combined_data.get("gender_male", True), True)],
        "bmi": [_safe_float(combined_data.get("bmi", 25.0), 25.0)],
        "sod": [_safe_bool_int(combined_data.get("sod", False), False)],
        "history_of_pep": [_safe_bool_int(combined_data.get("history_of_pep", False), False)],
        "hx_of_recurrent_pancreatitis": [_safe_bool_int(combined_data.get("hx_of_recurrent_pancreatitis", False), False)],
        "pancreatic_sphincterotomy": [_safe_bool_int(combined_data.get("pancreatic_sphincterotomy", False), False)],
        "precut_sphincterotomy": [_safe_bool_int(combined_data.get("precut_sphincterotomy", False), False)],
        "minor_papilla_sphincterotomy": [_safe_bool_int(combined_data.get("minor_papilla_sphincterotomy", False), False)],
        "failed_cannulation": [_safe_bool_int(combined_data.get("failed_cannulation", False), False)],
        "difficult_cannulation": [_safe_bool_int(combined_data.get("difficult_cannulation", False), False)],
        "pneumatic_dilation_of_intact_biliary_sphincter": [_safe_bool_int(combined_data.get("pneumatic_dilation_of_intact_biliary_sphincter", False), False)],
        "pancreatic_duct_injection": [_safe_bool_int(combined_data.get("pancreatic_duct_injections", False), False)],
        "pancreatic_duct_injections_2": [_safe_int(combined_data.get("pancreatic_duct_injections_2", 0), 0)],
        "acinarization": [_safe_bool_int(combined_data.get("acinarization", False), False)],
        "trainee_involvement": [_safe_bool_int(combined_data.get("trainee_involvement", False), False)],
        "cholecystectomy": [_safe_bool_int(combined_data.get("cholecystectomy", False), False)],
        "pancreo_biliary_malignancy": [_safe_bool_int(combined_data.get("pancreo_biliary_malignancy", False), False)],
        "guidewire_cannulation": [_safe_bool_int(combined_data.get("guidewire_cannulation", False), False)],
        "guidewire_passage_into_pancreatic_duct": [_safe_bool_int(combined_data.get("guidewire_passage_into_pancreatic_duct", False), False)],
        "guidewire_passage_into_pancreatic_duct_2": [_safe_int(combined_data.get("guidewire_passage_into_pancreatic_duct_2", 0), 0)],
        "biliary_sphincterotomy": [_safe_bool_int(combined_data.get("biliary_sphincterotomy", False), False)],
    })
    
    return df


def categorize_risk(risk_percentage: float) -> str:
    """Categorize risk percentage into low/moderate/high"""
    if risk_percentage < RISK_THRESHOLDS["low"]:
        return "low"
    elif risk_percentage < RISK_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "high"


def extract_r_model_results(r_output: Dict) -> tuple:
    """
    Extract and format results from R model output.
    R model returns:
        - final_risk: float (percentage, e.g., 10.7 for 10.7%)
        - reference_samples: DataFrame with similar cases
        - test_patient_prediction: DataFrame with ALL treatment predictions
        - explaination_text: DataFrame with feature contributions
    
    Args:
        r_output: Dict with R model output
    
    Returns:
        Tuple of (risk_score, risk_category, treatment_predictions, formatted_details)
    """
    # Extract final_risk (main prediction as percentage)
    final_risk = r_output.get("final_risk")
    if final_risk is None:
        raise ValueError("R model did not return final_risk")
    
    # Convert to float
    risk_score = float(final_risk)
    risk_category = categorize_risk(risk_score)
    
    # Extract ALL treatment scenario predictions
    treatment_predictions = []
    test_pred_key = "test_patient_prediction"
    
    if test_pred_key in r_output:
        test_pred = r_output[test_pred_key]
        
        # Manual conversion from R df to pandas df; more reliable than pandas2ri for our use case
        try:
            # If the bridge already converted to a pandas DataFrame, use it
            if isinstance(test_pred, pd.DataFrame):
                test_pred_df = test_pred.copy()
            # If it's an rpy2 object (ListVector / DataFrame-like), extract columns
            elif getattr(test_pred, 'names', None) is not None:
                test_pred_df = pd.DataFrame({
                    str(col): list(test_pred.rx2(str(col))) 
                    for col in test_pred.names
                })
            # If it's a mapping/OrderedDict or list-like, try to convert directly
            elif isinstance(test_pred, dict) or hasattr(test_pred, 'items'):
                test_pred_df = pd.DataFrame(test_pred)
            else:
                # Fallback: try pandas2ri to convert whatever rpy2-like object it is
                from rpy2.robjects import pandas2ri
                test_pred_df = pandas2ri.rpy2py(test_pred)
            print(f"Converted R DataFrame to pandas. Shape: {test_pred_df.shape}")
        except Exception as e:
            print(f"Error converting R DataFrame: {e}")
            test_pred_df = pd.DataFrame()
        
        if isinstance(test_pred_df, pd.DataFrame) and len(test_pred_df) > 0:
            for _, row in test_pred_df.iterrows():
                raw_therapy = str(row.get('therapy', 'No treatment'))
                pred_value = float(row.get('pred', 0)) * 100
                # map label -> canonical id + label
                tid = therapy_id_from_label(raw_therapy)
                tlabel = therapy_label_from_id(tid)

                treatment_predictions.append({
                    "therapy_id": tid.value,
                    "therapy_label": tlabel,
                    "risk_percentage": round(pred_value, 1),
                    "risk_category": categorize_risk(pred_value)
                })
            print(f"{len(treatment_predictions)} treatment predictions:")
            for tp in treatment_predictions:
                print(f"  - {tp['therapy_label']}: {tp['risk_percentage']}% ({tp['risk_category']})")
    
    # Format additional details from R model
    details = {
        "final_risk_percentage": risk_score,
        "model_version": "R_PEPRISC"
    }
    
    # Add reference samples summary if available
    if "reference_samples" in r_output:
        ref_samples = r_output["reference_samples"]
        if isinstance(ref_samples, pd.DataFrame) and len(ref_samples) > 0:
            details["reference_samples"] = {
                "count": len(ref_samples),
                "mean_risk": float(ref_samples['pred'].mean() * 100) if 'pred' in ref_samples.columns else None,
                "pep_cases": int(ref_samples['pep'].sum()) if 'pep' in ref_samples.columns else None,
                "sample_predictions": ref_samples[['patient_id', 'pred', 'pep']].head(5).to_dict('records') if 'pred' in ref_samples.columns else []
            }
    
    # Add feature importance/explanation if available
    if "explaination_text" in r_output:
        explanation = r_output["explaination_text"]
        if isinstance(explanation, pd.DataFrame) and len(explanation) > 0:
            if 'feature' in explanation.columns and 'feature_weight' in explanation.columns:
                top_features = explanation[['feature', 'feature_weight', 'feature_value']].drop_duplicates('feature').head(10)
                details["top_contributing_features"] = top_features.to_dict('records')
    
    return risk_score, risk_category, treatment_predictions, details


def predict_pep_risk(
    manual_data: Optional[Dict] = None,
    llm_extracted_data: Optional[Dict] = None
) -> Dict:
    """
    Predict PEP risk using the R PEPRISC model.
    
    Args:
        manual_data: Dict with manually input risk factors
        llm_extracted_data: Dict with LLM-extracted risk factors from transcript
    
    Returns:
        Dict with prediction results:
        {
            "success": bool,
            "risk_score": float (percentage),
            "risk_category": str ("low"/"moderate"/"high"),
            "model": str ("R_PEPRISC"),
            "combined_risk_factors": dict,
            "r_model_details": dict with R model output
        }
    """
    
    # Combine LLM-extracted data and manual data (manual should override LLM)
    combined_data = {}

    if llm_extracted_data:
        # Remove metadata fields from LLM extraction
        filtered_llm_data = {k: v for k, v in llm_extracted_data.items() if k not in ['id', 'model']}
        combined_data.update(filtered_llm_data)

    if manual_data:
        overlap = set(combined_data.keys()) & set(manual_data.keys())
        if overlap:
            print(f"Note: manual data will override LLM for fields: {sorted(list(overlap))}")
        combined_data.update(manual_data)
    
    try:
        # Load R model
        r_model = get_r_model()
        
        # Prepare input DataFrame for R model
        input_df = prepare_r_input_dataframe(combined_data)
        
        # Call R model (conversion context handled in PepriscBridge)
        print("Calling R PEPRISC model...")
        r_output = r_model.peprisk_predict(input_df)
        
        # Convert R output to a Python dict depending on the returned type
        r_dict = {}
        # This should be the case. bridge already returned a Python dict/OrderedDict
        if isinstance(r_output, dict) or hasattr(r_output, 'items'):
            r_dict = dict(r_output)

        print(f"R model returned: {list(r_dict.keys())}")
        
        # Extract and format results (including all treatment predictions)
        risk_score, risk_category, treatment_predictions, r_details = extract_r_model_results(r_dict)
        
        # Prepare response
        return {
            "success": True,
            "risk_score": round(risk_score, 2),
            "risk_category": risk_category,
            "treatment_predictions": treatment_predictions,
            "model": "R_PEPRISC",
            "combined_risk_factors": combined_data,
            "r_model_details": r_details
        }
        
    except Exception as e:
        error_msg = f"R model prediction failed: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": error_msg,
            "risk_score": None,
            "risk_category": None,
            "model": "R_PEPRISC"
        }


if __name__ == "__main__":
    # Test the R PEPRISC model with example cases
    print("=" * 70)
    print("Testing R PEPRISC Model Integration")
    print("=" * 70)
    print()
    
    # Test Case 1: Low risk patient
    print("Test Case 1: Low Risk Patient")
    print("-" * 70)
    manual_low = {
        "age_years": 55,
        "gender_male": True,
        "bmi": 28.2,
        "sod": False,
        "history_of_pep": False,
        "hx_of_recurrent_pancreatitis": False,
        "cholecystectomy": True,
        "trainee_involvement": False
    }
    
    llm_low = {
        "precut_sphincterotomy": False,
        "minor_papilla_sphincterotomy": False,
        "failed_cannulation": False,
        "difficult_cannulation": False,
        "pneumatic_dilation_of_intact_biliary_sphincter": False,
        "pancreatic_duct_injections": False,
        "pancreatic_duct_injections_2": 0,
        "acinarization": False,
        "pancreo_biliary_malignancy": False,
        "guidewire_cannulation": True,
        "guidewire_passage_into_pancreatic_duct": False,
        "guidewire_passage_into_pancreatic_duct_2": 0,
        "biliary_sphincterotomy": True
    }
    
    result_low = predict_pep_risk(manual_low, llm_low)
    print(f"Success: {result_low['success']}")
    print(f"Risk Score: {result_low['risk_score']}%")
    print(f"Risk Category: {result_low['risk_category']}")
    if result_low.get('r_model_details'):
        print(f"Reference samples: {result_low['r_model_details'].get('reference_samples', {}).get('count', 'N/A')}")
    print()
    
    # Test Case 2: High risk patient
    print("Test Case 2: High Risk Patient")
    print("-" * 70)
    manual_high = {
        "age_years": 35,
        "gender_male": False,
        "bmi": 22.0,
        "sod": True,
        "history_of_pep": True,
        "hx_of_recurrent_pancreatitis": False,
        "cholecystectomy": False,
        "trainee_involvement": True
    }
    
    llm_high = {
        "precut_sphincterotomy": True,
        "minor_papilla_sphincterotomy": False,
        "failed_cannulation": True,
        "difficult_cannulation": True,
        "pneumatic_dilation_of_intact_biliary_sphincter": False,
        "pancreatic_duct_injections": True,
        "pancreatic_duct_injections_2": 3,
        "acinarization": True,
        "pancreo_biliary_malignancy": False,
        "guidewire_cannulation": True,
        "guidewire_passage_into_pancreatic_duct": True,
        "guidewire_passage_into_pancreatic_duct_2": 2,
        "biliary_sphincterotomy": False
    }
    
    result_high = predict_pep_risk(manual_high, llm_high)
    print(f"Success: {result_high['success']}")
    print(f"Risk Score: {result_high['risk_score']}%")
    print(f"Risk Category: {result_high['risk_category']}")
    if result_high.get('r_model_details'):
        print(f"Reference samples: {result_high['r_model_details'].get('reference_samples', {}).get('count', 'N/A')}")
    print()
    
    print("=" * 70)
    print("Testing Complete")
    print("=" * 70)

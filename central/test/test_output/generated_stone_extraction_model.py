from typing import List, Optional, Union
from pydantic import BaseModel, field_validator

class StoneExtractionData(BaseModel):
    stone_extraction_performed: Optional[bool] = None
    stone_indications: Optional[str] = None
    initial_method: Optional[str] = None
    initial_result_text: Optional[str] = None
    eplbd_performed: Optional[bool] = None
    eplbd_sphinc: Optional[str] = None
    eplbd_balloon_type: Optional[str] = None
    eplbd_target_diameter: Optional[int] = None
    eplbd_inflation_strategy: Optional[str] = None
    eplbd_duration: Optional[int] = None
    eplbd_waist_resolution: Optional[str] = None
    eplbd_outcome: Optional[str] = None
    eplbd_adverse_events: Optional[str] = None
    mech_lithotripsy_performed: Optional[bool] = None
    mech_lithotripsy_indication: Optional[str] = None
    mech_lithotripsy_basket_system: Optional[str] = None
    lithotripsy_location: Optional[str] = None
    mech_lithotripsy_stone: Optional[str] = None
    mech_lithotripsy_outcome: Optional[str] = None
    mech_lithotripsy_basket_impaction: Optional[str] = None
    mech_lithotripsy_adverse_events: Optional[str] = None
    chol_lithotripsy_performed: Optional[bool] = None
    chol_lithotripsy_platform: Optional[str] = None
    chol_lithotripsy_modality: Optional[str] = None
    chol_lithotripsy_target_stones: Optional[str] = None
    chol_lithotripsy_irrigation: Optional[bool] = None
    chol_lithotripsy_fragmentation: Optional[str] = None
    chol_lithotripsy_debris_clearance: Optional[str] = None
    chol_lithotripsy_details: Optional[str] = None
    stone_therapy_limitations: Optional[str] = None
    duct_clearance_status: Optional[str] = None
    duct_clearance_narrative: Optional[str] = None
    staged_strategy_narrative: Optional[str] = None
    stone_therapy_complications: Optional[str] = None
    stone_therapy_bile_drainage_narrative: Optional[str] = None
    fluoroscopic_img_obtained: Optional[bool] = None

    @field_validator('eplbd_target_diameter', 'eplbd_duration', mode='before')
    @classmethod
    def coerce_integer_with_unknown(cls, v):
        """Handle 'unknown' sentinel for integer fields."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Treat unknown/none as None
            if v_lower in ('unknown', 'none', 'n/a', 'na', '-1'):
                return None
            # Try to parse as integer
            try:
                return int(v)
            except ValueError:
                return None
        return v

    @field_validator('stone_extraction_performed', 'eplbd_performed', 'mech_lithotripsy_performed', 'chol_lithotripsy_performed', 'chol_lithotripsy_irrigation', 'fluoroscopic_img_obtained', mode='before')
    @classmethod
    def coerce_boolean_with_unknown(cls, v):
        """Handle 'unknown' sentinel for boolean fields and coerce truthy/falsey strings."""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Treat unknown/none as None
            if v_lower in ('unknown', 'none', 'n/a', 'na', '-1'):
                return None
            # Coerce truthy/falsey strings
            if v_lower in ('true', 'yes', '1'):
                return True
            if v_lower in ('false', 'no', '0'):
                return False
        # Pass through as-is and let pydantic handle validation
        return v
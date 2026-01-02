from typing import List, Optional, Union
from pydantic import BaseModel, field_validator

class DifficultCannulationData(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    chief_complaints: Optional[str] = None
    symptoms_duration: Optional[str] = None
    symptoms_narrative: Optional[str] = None
    negative_history: Optional[str] = None
    past_medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    medications: Optional[str] = None
    monitoring: Optional[str] = None
    duodenoscope_type: Optional[str] = None
    grade_of_ercp: Optional[int] = None
    pd_cannulation_status: Optional[str] = None
    cannulation_success: Optional[bool] = None
    lactated_ringers: Optional[bool] = None
    rectal_indomethacin: Optional[bool] = None
    successful_completion: Optional[bool] = None
    failed_ercp: Optional[bool] = None
    scout_film_status: Optional[str] = None
    scout_film_optional_findings: Optional[str] = None
    scout_film_free_text: Optional[str] = None
    scope_advance_difficulty: Optional[str] = None
    upper_gi_examination_extent: Optional[str] = None
    upper_gi_findings: Optional[str] = None
    major_papilla_status: Optional[str] = None
    major_papilla_abnormal_morphology: Optional[str] = None
    prior_biliary_sphincterotomy_evidence: Optional[bool] = None
    prior_biliary_sphincterotomy_orifice_patency: Optional[str] = None
    papilla_stent_present: Optional[bool] = None
    periampullary_diverticulum_present: Optional[bool] = None
    papilla_diverticulum_loc: Optional[str] = None
    major_papilla_free_text: Optional[str] = None
    minor_papilla_status: Optional[str] = None
    minor_papilla_morphology: Optional[str] = None
    minor_papilla_prior_sphincterotomy: Optional[bool] = None
    minor_papilla_orifice_patency: Optional[str] = None
    minor_papilla_free_text: Optional[str] = None
    ampulla_overall_appearance: Optional[str] = None
    ampulla_pus_present: Optional[bool] = None
    ampulla_active_bleeding: Optional[bool] = None
    ampulla_free_text: Optional[str] = None
    ampullectomy_performed: Optional[bool] = None
    snare_size_mm: Optional[int] = None
    resection_style: Optional[str] = None
    energy_settings: Optional[str] = None
    specimen_retrieval_method: Optional[str] = None
    resection_base_assessment: Optional[str] = None
    ampullectomy_hemostasis_method: Optional[str] = None
    sphincterotome_type: Optional[str] = None
    guidewire_type: Optional[str] = None
    bile_duct_cannulation_successful: Optional[bool] = None
    bile_duct_cannulation_difficulty: Optional[str] = None
    advanced_cannulation_techniques: Optional[str] = None
    rendezvous_attempted: Optional[bool] = None
    rendezvous_result: Optional[str] = None
    pancreatic_duct_cannulation_attempted: Optional[str] = None
    pancreatic_duct_cannulation_success: Optional[bool] = None
    pancreatic_duct_cannulation_technique: Optional[str] = None
    biliary_rendezvous_route: Optional[str] = None
    contrast_injection_performed: Optional[bool] = None
    cbd_diameter_mm: Optional[float] = None
    ihd_status: Optional[str] = None
    stone_description: Optional[str] = None
    stricture_description: Optional[str] = None
    non_stone_filling_defects: Optional[str] = None
    bile_leak_location: Optional[str] = None
    prior_stent_status: Optional[str] = None
    cholangiogram_free_text: Optional[str] = None
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
    pancreatogram_obtained: Optional[bool] = None
    pancreatogram_overall: Optional[str] = None
    main_pd_diameter_head_mm: Optional[float] = None
    main_pd_diameter_body_mm: Optional[float] = None
    main_pd_diameter_tail_mm: Optional[float] = None
    side_branch_status: Optional[str] = None
    pd_stricture_location: Optional[str] = None
    pd_pseudocyst_communication: Optional[bool] = None
    sphincterotomy_type: Optional[str] = None
    sphincterotome_used: Optional[str] = None
    incision_direction: Optional[str] = None
    sphincterotomy_extent: Optional[str] = None
    sphincterotomy_goal_achieved: Optional[str] = None
    biliary_stent_type: Optional[str] = None
    plastic_biliary_size: Optional[str] = None
    plastic_biliary_type: Optional[str] = None
    plastic_biliary_length: Optional[int] = None
    metal_biliary_type: Optional[str] = None
    metal_biliary_type_features: Optional[str] = None
    metal_biliary_narrative: Optional[str] = None
    metal_biliary_length: Optional[int] = None
    pd_stent_placed: Optional[bool] = None
    pd_stent_purpose: Optional[str] = None
    pd_stent_size: Optional[str] = None
    pd_stent_type: Optional[str] = None
    pd_stent_therapy_flaps: Optional[str] = None
    pd_stent_therapy_narrative: Optional[str] = None
    pd_placement_narrative: Optional[str] = None
    estimated_blood_loss: Optional[float] = None
    specimens_removed: Optional[str] = None
    complications: Optional[str] = None
    impressions: Optional[List[str]] = None
    recommendations: Optional[str] = None

    @field_validator('age', 'grade_of_ercp', 'snare_size_mm', 'eplbd_target_diameter', 'eplbd_duration', 'plastic_biliary_length', 'metal_biliary_length', mode='before')
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

    @field_validator('cbd_diameter_mm', 'main_pd_diameter_head_mm', 'main_pd_diameter_body_mm', 'main_pd_diameter_tail_mm', 'estimated_blood_loss', mode='before')
    @classmethod
    def coerce_float_with_unknown(cls, v):
        """Handle 'unknown' sentinel for float fields."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Treat unknown/none as None
            if v_lower in ('unknown', 'none', 'n/a', 'na', '-1'):
                return None
            # Try to parse as float
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator('cannulation_success', 'lactated_ringers', 'rectal_indomethacin', 'successful_completion', 'failed_ercp', 'prior_biliary_sphincterotomy_evidence', 'papilla_stent_present', 'periampullary_diverticulum_present', 'minor_papilla_prior_sphincterotomy', 'ampulla_pus_present', 'ampulla_active_bleeding', 'ampullectomy_performed', 'bile_duct_cannulation_successful', 'rendezvous_attempted', 'pancreatic_duct_cannulation_success', 'contrast_injection_performed', 'stone_extraction_performed', 'eplbd_performed', 'mech_lithotripsy_performed', 'chol_lithotripsy_performed', 'chol_lithotripsy_irrigation', 'fluoroscopic_img_obtained', 'pancreatogram_obtained', 'pd_pseudocyst_communication', 'pd_stent_placed', mode='before')
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
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator

''' Validation model for LLM-extracted data.'''

class ColonoscopyData(BaseModel):
    indications: Optional[str] = None
    last_colonoscopy: Optional[str] = None
    bbps_simple: Optional[str] = None
    bbps_right: Optional[int] = None
    bbps_transverse: Optional[int] = None
    bbps_left: Optional[int] = None
    bbps_total: Optional[int] = None
    extent: Optional[str] = None
    findings: Optional[str] = None
    polyp_count: Optional[int] = None
    impressions: Optional[List[str]] = None

    @field_validator('bbps_right', 'bbps_transverse', 'bbps_left', 'bbps_total', mode='before')
    @classmethod
    def convert_na_to_none(cls, v):
        """Convert 'N/A' strings to None for optional integer fields"""
        if isinstance(v, str) and v.strip().upper() in ['N/A', 'NA']:
            return None
        return v

    @field_validator('impressions', mode='before')
    @classmethod
    def convert_na_to_empty_list(cls, v):
        """Convert 'N/A' strings to empty list for impressions, and strings to single-element lists"""
        if isinstance(v, str):
            if v.strip().upper() in ['N/A', 'NA']:
                return []
            # Convert any other string to a single-element list
            return [v]
        return v

class PolypData(BaseModel):
    size_min_mm: Optional[float] = None
    size_max_mm: Optional[float] = None
    location: Optional[str] = None
    resection_performed: Optional[bool] = None
    resection_method: Optional[str] = None
    nice_class: Optional[int] = None
    jnet_class: Optional[str] = None
    paris_class: Optional[str] = None

class EUSData(BaseModel):
    indications: Optional[str] = None
    samples_taken: Optional[bool] = None
    eus_findings: Optional[str] = None
    egd_findings: Optional[str] = None
    impressions: Optional[List[str]] = None

    @field_validator('impressions', mode='before')
    @classmethod
    def convert_impressions_to_list(cls, v):
        """Convert 'N/A' strings to empty list for impressions, and strings to single-element lists"""
        if isinstance(v, str):
            if v.strip().upper() in ['N/A', 'NA']:
                return []
            return [v]
        return v

class ERCPData(BaseModel):
    indications: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    chief_complaints: Optional[str] = None
    symptoms_duration: Optional[str] = None
    symptoms_description: Optional[str] = None
    negative_history: Optional[str] = None
    past_medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None
    duodenoscope_type: Optional[str] = None
    scout_film_status: Optional[str] = None
    scout_film_findings: Optional[str] = None
    scope_advancement_difficulty: Optional[str] = None
    upper_gi_examination: Optional[str] = None
    upper_gi_findings: Optional[str] = None
    grade_of_ercp: Optional[str] = None
    pd_cannulation: Optional[str] = None
    cannulation_success: Optional[bool] = None
    lactated_ringers: Optional[bool] = None
    rectal_indomethacin: Optional[bool] = None
    successful_completion_of_intended_procedure: Optional[bool] = None
    failed_ercp_from_another_facility_or_provider: Optional[bool] = None
    samples_taken: Optional[bool] = None
    egd_findings: Optional[str] = None
    ercp_findings: Optional[str] = None
    biliary_stent_type: Optional[str] = None
    pd_stent: Optional[bool] = None
    impressions: Optional[List[str]] = None

    @field_validator('impressions', mode='before')
    @classmethod
    def convert_impressions_to_list(cls, v):
        """Convert 'N/A' strings to empty list for impressions, and strings to single-element lists"""
        if isinstance(v, str):
            if v.strip().upper() in ['N/A', 'NA']:
                return []
            return [v]
        return v

class PEPRiskData(BaseModel):
    # sod: bool
    # history_of_pep: bool
    # hx_of_recurrent_pancreatitis: bool
    pancreatic_sphincterotomy: Optional[bool] = None
    precut_sphincterotomy: Optional[bool] = None
    minor_papilla_sphincterotomy: Optional[bool] = None
    failed_cannulation: Optional[bool] = None
    difficult_cannulation: Optional[bool] = None
    pneumatic_dilation_of_intact_biliary_sphincter: Optional[bool] = None
    pancreatic_duct_injections: Optional[bool] = None
    pancreatic_duct_injections_2: Optional[int] = None
    acinarization: Optional[bool] = None
    # trainee_involvement: bool
    # cholecystectomy: bool # (history, redcap)
    pancreo_biliary_malignancy: Optional[bool] = None
    guidewire_cannulation: Optional[bool] = None
    guidewire_passage_into_pancreatic_duct: Optional[bool] = None
    guidewire_passage_into_pancreatic_duct_2: Optional[int] = None
    biliary_sphincterotomy: Optional[bool] = None
    indomethacin_nsaid_prophylaxis: Optional[bool] = None
    aggressive_hydration: Optional[bool] = None
    pancreatic_duct_stent_placement: Optional[bool] = None

    # convert "N/A" to false
    @field_validator('*', mode='before')
    @classmethod
    def convert_na_to_false(cls, v):
        """Convert 'N/A' strings to False for optional boolean fields"""
        if isinstance(v, str) and v.strip().upper() in ['N/A', 'NA']:
            return False
        return v

class EGDData(BaseModel):
    indications: Optional[str] = None
    extent: Optional[str] = None
    samples_taken: Optional[bool] = None
    barrets_ablation: Optional[bool] = None
    bleeding_treatment: Optional[bool] = None
    peg_pej: Optional[bool] = None
    esophagus: Optional[str] = None
    stomach: Optional[str] = None
    duodenum: Optional[str] = None
    egd_findings: Optional[str] = None
    impressions: Optional[List[str]] = None

    @field_validator('impressions', mode='before')
    @classmethod
    def convert_impressions_to_list(cls, v):
        """Convert 'N/A' strings to empty list for impressions, and strings to single-element lists"""
        if isinstance(v, str):
            if v.strip().upper() in ['N/A', 'NA']:
                return []
            return [v]
        return v
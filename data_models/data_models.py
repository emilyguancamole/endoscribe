from typing import List, Optional, Union
from pydantic import BaseModel, Field

''' Validation model for LLM-extracted data.'''

class ColonoscopyData(BaseModel):
    indications: str
    last_colonoscopy: str #? optional?
    bbps_simple:str #? maybe recall model if missing. but this is safer
    bbps_right: Optional[int] 
    bbps_transverse: Optional[int]
    bbps_left: Optional[int]
    bbps_total: Optional[int]
    extent: str
    findings: str
    polyp_count: int
    impressions: List[str]

class PolypData(BaseModel):
    size_min_mm: float
    size_max_mm: float
    location: str
    resection_performed: bool
    resection_method: str
    nice_class: Optional[int]
    jnet_class: Optional[str]
    paris_class: Optional[str]

class EUSData(BaseModel):
    indications: str
    samples_taken: bool
    eus_findings: str
    egd_findings: str
    impressions: List[str]

class ERCPData(BaseModel):
    indications: str
    samples_taken: bool
    egd_findings: str
    ercp_findings: str
    biliary_stent_type: str
    pd_stent: bool
    impressions: List[str]

class PEPRiskData(BaseModel):
    # sod: bool
    # history_of_pep: bool
    # hx_of_recurrent_pancreatitis: bool
    pancreatic_sphincterotomy: bool
    precut_sphincterotomy: bool
    minor_papilla_sphincterotomy: bool
    failed_cannulation: bool
    difficult_cannulation: bool
    pneumatic_dilation_of_intact_biliary_sphincter: bool
    pancreatic_duct_injections: bool
    pancreatic_duct_injections_2: int
    acinarization: bool
    # trainee_involvement: bool
    # cholecystectomy: bool # (history, redcap)
    pancreo_biliary_malignancy: bool
    guidewire_cannulation: bool
    guidewire_passage_into_pancreatic_duct: bool
    guidewire_passage_into_pancreatic_duct_2: int
    biliary_sphincterotomy: bool
    indomethacin_nsaid_prophylaxis: bool
    aggressive_hydration: bool
    pancreatic_duct_stent_placement: bool

class EGDData(BaseModel):
    indications: str
    extent: str
    samples_taken: bool
    barrets_ablation: bool
    bleeding_treatment: bool
    peg_pej: bool
    esophagus: str
    stomach: str
    duodenum: str
    egd_findings: str
    impressions: List[str]
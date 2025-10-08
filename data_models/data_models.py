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
    egd_findings: str
    ercp_findings: str
    samples_taken: bool
    impressions: List[str]

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
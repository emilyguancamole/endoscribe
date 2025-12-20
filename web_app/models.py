from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ProcedureType(str, Enum):
    """Supported procedure types"""
    COL = "col"
    EUS = "eus"
    ERCP = "ercp"
    EGD = "egd"
    PEP_RISK = "pep_risk"


class TranscriptionChunk(BaseModel):
    """Real-time transcription chunk from WebSocket"""
    session_id: str
    text: str
    timestamp: float
    is_final: bool = False


class PEPRiskManualInput(BaseModel):
    """Manual PEP risk factors input by clinician"""
    age_years: int
    gender_male: bool
    bmi: float
    cholecystectomy: bool
    history_of_pep: bool
    hx_of_recurrent_pancreatitis: bool
    sod: bool
    pancreo_biliary_malignancy: bool
    trainee_involvement: bool


class ProcessRequest(BaseModel):
    """Request to process a transcript and extract structured data"""
    transcript: str = Field(..., min_length=10, description="The full transcript text to process")
    procedure_type: ProcedureType = Field(..., description="Type of procedure (col, eus, ercp, egd)")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
    manual_pep_data: Optional[PEPRiskManualInput] = Field(None, description="Manual PEP risk factors for ERCP procedures")

    @field_validator('transcript')
    @classmethod
    def validate_transcript(cls, v: str) -> str:
        """Ensure transcript is not just whitespace"""
        if not v.strip():
            raise ValueError("Transcript cannot be empty or whitespace only")
        return v.strip()


class ProcessResponse(BaseModel):
    """Response from processing endpoint"""
    success: bool
    procedure_type: str
    session_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict, description="Extracted structured data")
    error: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    pep_risk_score: Optional[float] = Field(None, description="PEP risk prediction score (0-100)")
    pep_risk_category: Optional[str] = Field(None, description="PEP risk category (low/moderate/high)")
    
    #?? additional fields for frontend compatibility
    colonoscopy_data: Optional[Dict[str, Any]] = Field(None, description="Colonoscopy-specific data")
    polyps_data: Optional[List[Dict[str, Any]]] = Field(None, description="Polyp data for colonoscopy")
    procedure_data: Optional[Dict[str, Any]] = Field(None, description="General procedure data (EUS, ERCP, EGD)")
    pep_risk_data: Optional[Dict[str, Any]] = Field(None, description="PEP risk assessment data")
    raw_output: Optional[str] = Field(None, description="Raw LLM output")
    formatted_note: Optional[str] = Field(None, description="Formatted clinical note")
    
    @property
    def status(self) -> str:
        """Legacy compatibility property"""
        return "success" if self.success else "error"


class ColonoscopyResult(BaseModel):
    """Colonoscopy-specific results"""
    colonoscopy: Dict[str, Any]
    polyps: List[Dict[str, Any]]


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str  # 'audio', 'transcript', 'error', 'status'
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    whisper_loaded: bool
    llm_initialized: bool
    supported_procedures: List[str]
    transcription_service: Optional[str] = None
    transcription_ready: Optional[bool] = None

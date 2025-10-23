from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class ProcedureType(str, Enum):
    """Supported procedure types"""
    COL = "col"
    EUS = "eus"
    ERCP = "ercp"
    EGD = "egd"


class TranscriptionChunk(BaseModel):
    """Real-time transcription chunk from WebSocket"""
    session_id: str
    text: str
    timestamp: float
    is_final: bool = False


class ProcessRequest(BaseModel):
    """Request to process a transcript and extract structured data"""
    transcript: str = Field(..., min_length=10, description="The full transcript text to process")
    procedure_type: ProcedureType = Field(..., description="Type of procedure (col, eus, ercp, egd)")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")

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

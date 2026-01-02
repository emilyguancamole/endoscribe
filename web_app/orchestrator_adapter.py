from pathlib import Path
from typing import Dict, Any, Optional
import sys

_here = Path(__file__).parent
_root = _here.parent
sys.path.insert(0, str(_root))
from central.orchestration.orchestrator import Orchestrator
from llm.client import LLMClient
from web_app.models import ProcessRequest, ProcessResponse, ProcedureType


class OrchestratorAdapter:
    """
    Bridges web API with orchestrator pipeline.
    Converts ProcessRequest -> Orchestrator.process_transcript() -> ProcessResponse
    """
    def __init__(self, llm_client: Optional[LLMClient] = None, enable_multipass: bool = True):
        """
        Args:
            llm_client: Optional pre-configured LLM client (web app's shared instance)
            enable_multipass: Whether to use multi-pass extraction (default True)
        """
        self.orchestrator = Orchestrator(
            template_dir=_root / "prompts",
            llm_client=llm_client,
            enable_multipass=enable_multipass
        )
    
    def process_request(self, request: ProcessRequest) -> ProcessResponse:
        """
        Process a transcript using orchestrator.
        Args:
            request: ProcessRequest with transcript, procedure_type, session_id
        Returns:
            ProcessResponse with structured data and formatted note
        """
        import time
        start_time = time.time()
        
        try:
            # Run orchestrator pipeline
            result = self.orchestrator.process_transcript(request.transcript)
            classification = result['classification']
            extracted_data = result['extracted_data']
            final_note = result['final_note']
            
            # Convert Pydantic model to dict
            extracted_dict = extracted_data.model_dump() if extracted_data else {}
            
            # Response for web app
            response_data = {
                'success': True,
                'procedure_type': request.procedure_type.value,
                'session_id': request.session_id,
                'data': extracted_dict,
                'formatted_note': final_note,
                'processing_time_seconds': round(time.time() - start_time, 2),
                'classification': {
                    'procedure_type': classification.procedure_type,
                    'active_modules': classification.active_modules,
                    'reasoning': classification.reasoning
                }
            }
            
            # TEMP Add procedure-specific fields for backward compatibility
            if request.procedure_type == ProcedureType.ERCP:
                response_data['procedure_data'] = extracted_dict
                # PEP risk will be handled separately if needed
            elif request.procedure_type == ProcedureType.EUS:
                response_data['procedure_data'] = extracted_dict
            elif request.procedure_type == ProcedureType.EGD:
                response_data['procedure_data'] = extracted_dict
            elif request.procedure_type == ProcedureType.COL:
                # Colonoscopy has colonoscopy_data and polyps_data
                response_data['colonoscopy_data'] = extracted_dict
                response_data['polyps_data'] = []  # TODO: extract polyps separately or ** fully combine them **
            
            return ProcessResponse(**response_data)
            
        except Exception as e:
            import traceback
            error_msg = f"Orchestrator processing failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return ProcessResponse(
                success=False,
                procedure_type=request.procedure_type.value,
                session_id=request.session_id,
                data={},
                formatted_note=f"# Processing Error\n\n{error_msg}",
                processing_time_seconds=round(time.time() - start_time, 2)
            )
    
    def get_extracted_data_for_export(
        self, 
        session_data: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Extract the Pydantic model from session data for DOCX export.
        Args:
            session_data: Session dictionary with results
        Returns:
            Pydantic model or None if not available
        """
        try:
            results = session_data.get('results', {})
            data = results.get('data', {})
            if not data:
                return None
            
            procedure_type = results.get('procedure_type', 'ercp')
            if procedure_type == 'ercp':
                from models.generated_ercp_base_model import ErcpBaseData
                return ErcpBaseData(**data)
            elif procedure_type == 'eus':
                from models.data_models import EUSData
                return EUSData(**data)
            elif procedure_type == 'egd':
                from models.data_models import EGDData
                return EGDData(**data)
            elif procedure_type == 'col':
                from models.data_models import ColonoscopyData
                return ColonoscopyData(**data)
            else:
                return None
        
        except Exception as e:
            print(f"Failed to reconstruct model for export: {e}")
            return None

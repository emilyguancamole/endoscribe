from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel
import sys

_here = Path(__file__).parent
_root = _here.parent.parent
sys.path.insert(0, str(_root))

from central.classification.classifier import ProcedureClassifier
from central.orchestration.template_assembler import TemplateAssembler
from central.orchestration.extraction_planner import ExtractionPlanner
from central.extraction.field_extractor import FieldExtractor
from central.drafters.ercp import ERCPDrafter
from llm.client import LLMClient


class Orchestrator:
    """ End-to-end procedure processing pipeline
    - ProcedureClassifier - identifies applicable modules
    - TemplateAssembler - merges relevant YAML templates
    - ExtractionPlanner - plans extraction by grouping modules
    - FieldExtractor - generates models/prompts and extracts fields
    - Drafter - generates final note from extracted data
    """
    def __init__(
        self, 
        template_dir: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None,
        config_name: str = "openai_gpt4o",
        enable_multipass: bool = True
    ):
        """
        Args:
            template_dir: Root directory for YAML templates
            llm_client: Optional pre-configured LLMClient
            config_name: Config name to use if llm_client not provided
            enable_multipass: Whether to use multipass extraction (default True)
        """
        if template_dir is None:
            template_dir = _root / "templating" / "prompts"
        
        self.template_dir = Path(template_dir)
        self.llm_client = llm_client or LLMClient.from_config(config_name)
        self.classifier = ProcedureClassifier(llm_client=self.llm_client)
        self.assembler = TemplateAssembler(template_dir=self.template_dir)
        
        self.planner = ExtractionPlanner()
        self.enable_multipass = enable_multipass
    
    def process_transcript(
        self, 
        transcript: str,
    ) -> Dict[str, Any]:
        """
        Classify > assemble > extract > generate
        Args:
            transcript: Raw procedure transcript text
        Returns:
            Dict with keys: classification, merged_template, extraction_passes, extracted_data, final_note
        """
        
        print("Classifying procedure type...")
        classification = self.classifier.classify_procedure(transcript)
        print(f"   Classification: {classification}")
        
        print("\nAssembling template...")
        merged_template = self.assembler.assemble_template(classification)
        print(f"   Merged {len(merged_template.get('field_groups', {}))} field groups")
        self.assembler.validate_template(merged_template)
        
        # Extract fields
        print("\nExtracting fields...")
        extractor = FieldExtractor(
            merged_template,
            llm_client=self.llm_client,
            output_dir=_root / "results" / "generated_artifacts"
        )
        if self.enable_multipass:
            extraction_passes = self.planner.plan_extraction_passes(merged_template)
            print(f"   MULTIPASS: Planned {len(extraction_passes)} extraction passes:")
            for pass_obj in extraction_passes:
                field_count = self._count_fields(pass_obj.template)
                print(f"     - {pass_obj.group_name}: {field_count} fields")
            extracted_data = extractor.multipass_extraction(transcript, extraction_passes)
        else:
            print("   SINGLE-PASS extraction")
            extracted_data = extractor.extract_fields(transcript)
            extraction_passes = None
        print(f"    Extracted {len(extracted_data.model_dump())} fields")
        
        # Generate note using drafter
        print("\nGenerating note...")
        final_note = self.generate_note(merged_template, extracted_data)
        
        return {
            'classification': classification,
            'merged_template': merged_template,
            'extraction_passes': extraction_passes,
            'extracted_data': extracted_data,
            'final_note': final_note
        }
    
    
    
    
    def _count_fields(self, template: Dict[str, Any]) -> int:
        """Count total fields in a template"""
        count = 0
        for fg_config in template.get('field_groups', {}).values():
            count += len(fg_config.get('fields', []))
        return count
    
    def generate_note(
        self, 
        merged_template: Dict[str, Any],
        extracted_data: BaseModel,
        note_format: str = 'markdown'
    ) -> str:
        """
        Generate note from extracted data using drafter.
        Args:
            merged_template: Full merged template with all modules
            extracted_data: Validated Pydantic model with field data
            format: Output format ('markdown' or 'docx')
        Returns:
            Formatted note (string or Document)
        """
        try:
            # Get procedure type
            proc_type = merged_template.get('meta', {}).get('procedure_type', 'ercp')
            proc_group = merged_template.get('meta', {}).get('procedure_group', 'ercp')
            
            # Get drafter config path
            drafter_config_path = _root / "results" / "generated_artifacts" / f"generated_{proc_type}_drafter.yaml"
            if proc_group == 'ercp':
                drafter = ERCPDrafter(procedure_type=proc_type)
            else:
                #todo
                pass
            
            note = drafter.render(
                extracted_data=extracted_data,
                drafter_config_path=drafter_config_path,
                format=note_format
            )
            print(f"   Generated {note_format} note")
            
            return note
            
        except Exception as e:
            print(f"   Error generating note: {e}")
            import traceback
            traceback.print_exc()
            return f"# ERROR GENERATING NOTE\n\n{str(e)}"

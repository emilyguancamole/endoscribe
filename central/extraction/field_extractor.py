"""
FieldExtractor: Generates artifacts from merged YAML and extracts fields from transcript.

Handles:
1. Generate Pydantic model from merged YAML
2. Generate LLM extraction prompt from merged YAML
3. Generate drafter template from merged YAML
4. Call LLM to extract fields using generated prompt
5. Validate and parse LLM response into Pydantic model
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Type
import sys
from pydantic import BaseModel, ValidationError
import importlib.util

_here = Path(__file__).parent
_root = _here.parent.parent
sys.path.insert(0, str(_root))

from central.orchestration.extraction_planner import ExtractionPass
from llm.client import LLMClient
from templating.generate_from_fields import (
    generate_prompt_text,
    generate_pydantic_model,
    generate_base_yaml
)

class FieldExtractor:
    """Extracts fields from transcript with generated prompt, validates via generated Pydantic model"""
    
    def __init__(
        self,
        merged_template: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        config_name: str = "openai_gpt4o",
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            merged_template: Merged YAML configuration from TemplateAssembler
            llm_client: Optional pre-configured LLMClient
            config_name: Config name to use if llm_client not provided
            output_dir: Optional directory to write generated artifacts (for debugging)
        """
        self.merged_template = merged_template
        self.llm_client = llm_client or LLMClient.from_config(config_name)
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Generate artifacts from merged template
        self.prompt_text = self._generate_prompt()
        self.model_class = self._generate_model()
        self.drafter_yaml = self._generate_drafter_template()
        
        # Write artifacts for debugging
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._write_artifacts()
            
    def multipass_extraction(
        self, 
        transcript: str, 
        extraction_passes: List[ExtractionPass]
    ) -> BaseModel:
        """
        Execute multiple extraction passes and merge results
        Args:
            transcript: Raw procedure transcript
            extraction_passes: List of ExtractionPass objects
        Returns:
            Merged Pydantic model with all extracted fields
        """
        pass_results = {}
        
        # For each extraction pass
        for i, pass_obj in enumerate(extraction_passes):
            print(f"\n   EXTRACTION PASS {i+1}/{len(extraction_passes)}: {pass_obj.group_name}")
            extractor = FieldExtractor(
                pass_obj.template,
                llm_client=self.llm_client,
                output_dir=_root / "results" / "generated_artifacts"
            )
            result = extractor.extract_fields(transcript)
            pass_results[pass_obj.group_name] = result
            field_count = len(result.model_dump())
            print(f"     Extracted {field_count} fields from {pass_obj.group_name}")
        
        # Merge all results into a single model
        print("\n   Merging extraction results...")
        merged_data = self._merge_extraction_models(pass_results)
        
        return merged_data

    
    def extract_fields(self, transcript: str) -> BaseModel:
        """
        Extract fields from transcript via LLM
        Args:
            transcript: Raw procedure transcript text
        Returns:
            Pydantic model instance with extracted fields
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert medical transcription system. Extract structured data from procedure transcripts accurately and completely, following instructions."
            }, #TODO add few shot
            {
                "role": "user",
                "content": f"{self.prompt_text}\n\n###TRANSCRIPT###\n{transcript}"
            }
        ]
        response = self.llm_client.chat_llm(messages)
        extracted_data = self._validate_pydantic(response)
        return extracted_data
    
    def _merge_extraction_models(self, pass_results: Dict[str, BaseModel]) -> BaseModel:
        """
        Merge multiple Pydantic models from different extraction passes
        Args:
            pass_results: Dict mapping group_name -> extracted BaseModel
        Returns:
            Single merged BaseModel with all fields
        """
        # Combine all field data into a single dict
        merged_dict = {}
        for _, model in pass_results.items():
            merged_dict.update(model.model_dump())
        
        # Create a merged model class, using first model as the base
        model_classes = list(pass_results.values())
        if not model_classes:
            raise ValueError("No extraction results to merge")
        # For simplicity, generic container model for now
        from pydantic import create_model
        # Get all field definitions from all models
        field_definitions = {}
        for model in model_classes:
            for field_name, field_info in model.model_fields.items():
                field_definitions[field_name] = (field_info.annotation, field_info.default)
        MergedModel = create_model('MergedExtractionData', **field_definitions)
        return MergedModel(**merged_dict)
    
    
    def _generate_prompt(self) -> str:
        """Generate LLM extraction prompt from merged template"""
        return generate_prompt_text(self.merged_template)
    
    def _generate_model(self) -> Type[BaseModel]:
        """
        Generate and dynamically load Pydantic model from merged template
        Returns:
            Pydantic model class
        """
        proc_type = self.merged_template.get('meta', {}).get('procedure_type', 'procedure')
        proc_type = proc_type[0] if proc_type else 'procedure'
        
        proc_type_clean = self._clean_proc_name(proc_type)
        model_name = f"{proc_type_clean.replace('_', ' ').title().replace(' ', '')}Data"
        
        # Generate validation model
        model_code = generate_pydantic_model(self.merged_template, model_name)
        
        # Dynamically load the model class
        spec = importlib.util.spec_from_loader(f"dynamic_model_{proc_type}", loader=None, origin="dynamic")
        # Store in module object's namespace in memory only; lives as long as FieldExtractor instance lives
        module = importlib.util.module_from_spec(spec)
        # Execute model code in module namespace
        exec(model_code, module.__dict__)
        
        model_class = getattr(module, model_name)
        return model_class
    
    def _generate_drafter_template(self) -> str:
        """Generate drafter YAML template from merged template"""
        procedure_meta = self.merged_template.get('meta', {})
        return generate_base_yaml(self.merged_template, procedure_meta)
    
    def _validate_pydantic(self, response: Dict) -> BaseModel:
        """
        Validate LLM response against Pydantic model
        Args:
            response: json dictionary from LLM response
        Returns:
            Validated Pydantic model instance
        """
        try:
            return self.model_class(**response)
        except ValidationError as e:
            raise ValueError(f"Failed to validate extracted fields: {e}")
    
    def _clean_proc_name(self, proc_type: str) -> str:
        """Clean procedure type string for valid class name"""
        proc_type_clean = proc_type.replace('.', '_').replace('-', '_')
        if proc_type_clean and proc_type_clean[0].isdigit():
            proc_type_clean = f"Proc{proc_type_clean}"
        return proc_type_clean
    
    def _write_artifacts(self):
        """Write generated artifacts to disk for debugging"""
        proc_type = self.merged_template.get('meta', {}).get('procedure_type', 'procedure')
        
        if isinstance(proc_type, list):
            proc_type = proc_type[0] if proc_type else 'procedure'
        
        # Write prompt, model
        prompt_path = self.output_dir / f"generated_{proc_type}_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(self.prompt_text)
        
        proc_type_clean = self._clean_proc_name(proc_type)
        model_name = f"{proc_type_clean.replace('_', ' ').title().replace(' ', '')}Data"
        model_code = generate_pydantic_model(self.merged_template, model_name)
        model_path = self.output_dir / f"generated_{proc_type}_model.py"
        with open(model_path, 'w') as f:
            f.write(model_code)
        
        # Write drafter template
        drafter_path = self.output_dir / f"generated_{proc_type}_drafter.yaml"
        with open(drafter_path, 'w') as f:
            f.write(self.drafter_yaml)
        
        print(f"   Artifacts written to {self.output_dir}")
        print(f"     - Prompt: {prompt_path}")
        print(f"     - Model: {model_path}")
        print(f"     - Drafter YAML: {drafter_path}")

    
"""
TemplateAssembler: Loads and merges YAML templates based on classification.

1. Load base YAML template
2. Load module YAMLs based on active modules
3. Merge field groups while preserving template structure
4. Validate merged configuration
"""

from pathlib import Path
from typing import Dict, Optional, Any, List
from copy import deepcopy
import sys

_here = Path(__file__).parent
_root = _here.parent.parent
sys.path.insert(0, str(_root))

from central.classification.classifier import ProcedureClassification
from templating.generate_from_fields import load_fields_config, merge_configs


class TemplateAssembler:
    """
    Assembles YAML templates by merging base + modules
    Tracks extraction_group metadata during merging to enable multi-pass extraction.
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Args:
            template_dir: Root directory for YAML templates (defaults to prompts/)
        """
        if template_dir is None:
            template_dir = _root / "prompts"
        self.template_dir = Path(template_dir)
        self.base_template_cache = {}
        self.module_cache = {}
        self.extraction_group_map: Dict[str, str] = {}  # field_group_name -> extraction_group
    
    def assemble_template(self, classification: ProcedureClassification) -> Dict[str, Any]:
        """
        Merge yaml template based on classification
        Args:
            classification: ProcedureClassification with active_modules
        Returns:
            Merged yaml configuration dict with base + modules + metadata
        """
        self.extraction_group_map = {} # e.g. 'quality_metrics' -> 'base', 'stone_details' -> 'stone_management'
        base_path = self._resolve_base_template_path(classification.base_template)
        base_config = load_fields_config(str(base_path))
        
        # Track base template field_groups with their extraction_group
        base_extraction_group = base_config.get('meta', {}).get('extraction_group', 'base')
        for fg_name in base_config.get('field_groups', {}).keys():
            self.extraction_group_map[fg_name] = base_extraction_group
        
        # Merge sibling yaml files in the same folder that have insert_after
        # Allows files, e.g. history, to be added to base template without being listed as modules by classifier
        base_dir = base_path.parent
        for sibling in sorted(base_dir.glob("*.yaml")):
            if sibling.resolve() == base_path.resolve():
                continue # skip the base
            try:
                sibling_cfg = load_fields_config(str(sibling))
            except Exception:
                print(f"   ! Warning: Could not load sibling template {sibling}")
                import traceback
                traceback.print_exc()
                continue
            # Only consider siblings that explicitly request insertion
            if sibling_cfg.get('meta', {}).get('insert_after'):
                # track extraction_group for any new field_groups
                for fg_name in sibling_cfg.get('field_groups', {}).keys():
                    if fg_name not in self.extraction_group_map:
                        self.extraction_group_map[fg_name] = sibling_cfg.get('meta', {}).get('extraction_group', 'unknown')
                # merge into base_config
                base_config = merge_configs(base_config, sibling_cfg)
        
        # No modules, return base (+ siblings)
        if not classification.active_modules:
            base_config['meta']['extraction_group_map'] = self.extraction_group_map
            return base_config
        
        # Load and merge modules
        merged_config = deepcopy(base_config)
        for module_id in classification.active_modules:
            module_path = self._resolve_module_template_path(classification.base_template, module_id)
            if not module_path.exists():
                print(f"   ! Warning: Module {module_id} not found at {module_path}")
                continue
            module_config = load_fields_config(str(module_path))
            # Only track NEW field_groups from this module
            module_extraction_group = module_config.get('meta', {}).get('extraction_group', 'unknown')
            new_field_groups = set(module_config.get('field_groups', {}).keys()) - set(base_config.get('field_groups', {}).keys())
            for fg_name in new_field_groups:
                self.extraction_group_map[fg_name] = module_extraction_group
            
            merged_config = merge_configs(merged_config, module_config)
        
        merged_config.setdefault('meta', {})
        merged_config['meta']['active_modules'] = classification.active_modules
        # Store extraction_group_map in meta for use by ExtractionPlanner
        merged_config['meta']['extraction_group_map'] = self.extraction_group_map
        
        return merged_config
    
    def _resolve_base_template_path(self, base_template: str) -> Path:
        """
        Resolve path to base template YAML
        Args:
            base_template: Template identifier like "ercp/base"
        Returns:
            Full path to base YAML file
        """
        # "ercp/base" -> prompts/ercp/yaml/fields_base.yaml
        parts = base_template.split('/')
        if len(parts) == 2:
            proc_group, template_name = parts
            return self.template_dir / proc_group / "yaml" / f"fields_{template_name}.yaml"
        # Fallback: direct path
        return self.template_dir / f"{base_template}.yaml"
    
    def _resolve_module_template_path(self, base_template: str, module_id: str) -> Path:
        """
        Resolve module YAML path. Handle code-based (0.2) and name-based (stone_management) ids.
        Args:
            base_template: Base template identifier like "ercp/base"
            module_id: Module identifier like "0.2", "stone_management", "base"
        Returns:
            Full resolved path to module YAML
        """
        parts = base_template.split('/')
        if len(parts) == 2:
            proc_group = parts[0]
            modules_dir = self.template_dir / proc_group / "yaml" / "modules"
            # Skip "base"
            if module_id == "base":
                return self.template_dir / proc_group / "yaml" / f"fields_base.yaml"
            if modules_dir.exists():
                # try exact match by code
                matching_files = list(modules_dir.glob(f"{module_id}_*.yaml"))
                if matching_files:
                    return matching_files[0]
                
                # try finding by name
                for yaml_file in modules_dir.glob("*.yaml"):
                    if module_id in yaml_file.stem:
                        return yaml_file
            
            # Fallback: construct expected path
            return modules_dir / f"{module_id}.yaml"
        # Fallback: direct path
        return self.template_dir / "modules" / f"{module_id}.yaml"
    
    # Note: merging logic moved to templating.generate_from_fields._merge_configs
    
    def validate_template(self, config: Dict[str, Any]) -> bool:
        """
        Validate merged template configuration by checking required keys
        Args:
            config: Merged configuration dict
        Returns:
            True if valid, else ValueError
        """
        # Check required top-level keys
        required_keys = ['meta', 'field_groups']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        # Check meta has required fields
        meta = config['meta']
        required_meta = ['procedure_group']#, 'procedure_type']
        for key in required_meta:
            if key not in meta:
                raise ValueError(f"Missing required meta field: {key}")
        
    
    def get_extraction_groups(self, merged_template: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Get mapping of extraction_group -> list of field_group names
        Args:
            merged_template: Merged template from assemble_template()
        Returns:
            Dict mapping extraction_group name to list of field_group names in that group
        """
        extraction_group_map = merged_template.get('meta', {}).get('extraction_group_map', {})
        
        # Invert mapping: field_group -> extraction_group ==> extraction_group -> [field_groups]
        result = {}
        for fg_name, eg_name in extraction_group_map.items():
            if eg_name not in result:
                result[eg_name] = []
            result[eg_name].append(fg_name)
        
        return result
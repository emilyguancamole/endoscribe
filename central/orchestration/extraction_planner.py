from pathlib import Path
from typing import Dict, List, Any, Optional
from copy import deepcopy
import sys

_here = Path(__file__).parent
_root = _here.parent.parent
sys.path.insert(0, str(_root))


class ExtractionPass:
    """Represents a single extraction pass with its template and metadata"""
    
    def __init__(
        self,
        group_name: str,
        template: Dict[str, Any],
        module_ids: List[str],
        priority: int = 0
    ):
        """
        Args:
            group_name: Name of the extraction group (e.g., "base", "stone_management")
            template: YAML template config for this pass
            module_ids: List of module IDs included in this pass
            priority: Execution priority (lower = earlier, 0 = first)
        """
        self.group_name = group_name
        self.template = template
        self.module_ids = module_ids
        self.priority = priority

class ExtractionPlanner:
    """Plans multi-pass extraction by grouping modules"""
    # Priority order for extraction groups
    GROUP_PRIORITIES = {
        'history': 0,
        'base': 1,
        'access_intervention': 2,
        'stone_management': 3,
        'drainage_management': 4, 
        'diagnostic': 5, 
        'complications': 6,  
    }
    
    def __init__(self):
        pass
    
    def plan_extraction_passes(
        self, 
        merged_template: Dict[str, Any],
        max_fields_per_pass: Optional[int] = None
    ) -> List[ExtractionPass]:
        """
        Split merged template into extraction passes by group
        Args:
            merged_template: Fully merged YAML template with all modules
            max_fields_per_pass: Optional limit on fields per pass (for adaptive splitting)
        
        Returns:
            List of ExtractionPass objects in priority order
        """
        grouped_templates = self._group_by_extraction_group(merged_template)
        
        # Create ExtractionPass objects with priorities
        passes = []
        for group_name, group_template in grouped_templates.items():
            priority = self.GROUP_PRIORITIES.get(group_name, 99)
            module_ids = group_template.get('meta', {}).get('active_modules', [])
            
            passes.append(ExtractionPass(
                group_name=group_name,
                template=group_template,
                module_ids=module_ids,
                priority=priority
            ))
        
        # Sort by priority
        passes.sort(key=lambda p: p.priority)
        
        # Optional: Further split if any pass exceeds max_fields_per_pass
        if max_fields_per_pass:
            passes = self._adaptive_split(passes, max_fields_per_pass)
        
        return passes
    
    def _group_by_extraction_group(self, merged_template: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Split merged template into templates per extraction_group using extraction_group_map from TemplateAssembler
        Args:
            merged_template: merged template with extraction_group_map in meta
        Returns:
            Dict { group_name -> template for that group }
        """
        field_groups = merged_template.get('field_groups', {})
        meta = merged_template.get('meta', {})
        extraction_group_map = meta.get('extraction_group_map', {})
        
        if not extraction_group_map:
            print("   Warning: No extraction_group_map found, fallback to single-pass")
            return {'base': merged_template}
        
        # Group field_groups by their extraction_group
        grouped_field_groups = {}
        
        for fg_name, fg_config in field_groups.items():
            # Use authoritative map from TemplateAssembler
            group_name = extraction_group_map.get(fg_name, 'base')
            
            if group_name not in grouped_field_groups:
                grouped_field_groups[group_name] = {}
            
            grouped_field_groups[group_name][fg_name] = fg_config
        
        # Build separate templates for each group
        result = {} # group_name (e.g. 'base') -> template dict from yaml
        for group_name, group_fgs in grouped_field_groups.items():
            group_template = deepcopy(merged_template)
            group_template['field_groups'] = group_fgs
            group_template['meta']['extraction_group'] = group_name
            group_template['meta']['procedure_type'] = f"{meta.get('procedure_type', 'procedure')}_{group_name}"
            result[group_name] = group_template
        print("Resulting grouped templates, ExtractionPlanner:\n", result)
        
        return result
    
    def _adaptive_split(
        self, 
        passes: List[ExtractionPass], 
        max_fields: int
    ) -> List[ExtractionPass]:
        """
        Further split passes that exceed field count threshold
        Args:
            passes: List of extraction passes
            max_fields: Maximum fields per pass
        Returns:
            Updated list with large passes split
        """
        new_passes = []
        
        for pass_obj in passes:
            field_count = self._count_fields(pass_obj.template)
            
            if field_count <= max_fields:
                new_passes.append(pass_obj)
            else:
                # Split this pass into smaller chunks
                split_passes = self._split_pass(pass_obj, max_fields)
                new_passes.extend(split_passes)
        
        return new_passes
    
    def _count_fields(self, template: Dict[str, Any]) -> int:
        """Count total number of fields in template"""
        count = 0
        for fg_config in template.get('field_groups', {}).values():
            count += len(fg_config.get('fields', []))
        return count
    
    def _split_pass(
        self, 
        pass_obj: ExtractionPass, 
        max_fields: int
    ) -> List[ExtractionPass]:
        """
        Split a large pass into smaller chunks
        Args:
            pass_obj: ExtractionPass to split
            max_fields: Maximum fields per chunk
        Returns:
            List of smaller ExtractionPass objects
        """
        # Simple strategy: split field_groups into chunks
        field_groups = pass_obj.template.get('field_groups', {})
        chunks = []
        current_chunk = {}
        current_count = 0
        
        for fg_name, fg_config in field_groups.items():
            fg_field_count = len(fg_config.get('fields', []))
            
            if current_count + fg_field_count > max_fields and current_chunk:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = {}
                current_count = 0
            
            current_chunk[fg_name] = fg_config
            current_count += fg_field_count
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Create ExtractionPass for each chunk
        split_passes = []
        for i, chunk_fgs in enumerate(chunks):
            chunk_template = deepcopy(pass_obj.template)
            chunk_template['field_groups'] = chunk_fgs
            chunk_template['meta']['procedure_type'] = f"{pass_obj.group_name}_part{i+1}"
            
            split_passes.append(ExtractionPass(
                group_name=f"{pass_obj.group_name}_part{i+1}",
                template=chunk_template,
                module_ids=pass_obj.module_ids,
                priority=pass_obj.priority
            ))
        
        return split_passes

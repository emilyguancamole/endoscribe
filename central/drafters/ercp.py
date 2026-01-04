from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from docx import Document
from pydantic import BaseModel

from central.drafters.base import NoteDrafter
from templating.generate_from_fields import build_report_sections


class ERCPDrafter(NoteDrafter):    
    def __init__(
        self,
        template_config_path: Optional[Union[str, Path]] = None,
        procedure_type: str = "ercp"
    ):
        super().__init__(template_config_path, procedure_type)
    
    def get_section_order(self) -> List[str]:
        return [
            'indications',
            'history',
            'medications',
            'monitoring',
            'description_of_procedure',
            'ercp_quality_metrics',
            'findings',
            'impressions',
            'recommendations'
        ]
    
    def format_impressions(self, impressions: List) -> List[str]:
        if isinstance(impressions, (list, tuple)):
            return [str(x).strip() for x in impressions if str(x).strip()]
        return []
    
    def format_recommendations(
        self,
        recommendations: List,
        extracted_data: Dict[str, Any]
    ) -> List[str]:
        """Format recommendations and generate ERCP recommendations.
        Args:
            recommendations: list of existing recs
            extracted_data: Full extracted data for context
        Returns:
            List of formatted recommendation strings
        """
        rec = []
        # Existing recommendations
        if isinstance(recommendations, (list, tuple)):
            rec.extend([str(x).strip() for x in recommendations if str(x).strip()])
        
        # Generate recommendations
        rec_generated = self._generate_ercp_recommendations(extracted_data)
        rec.extend(rec_generated)
        
        # Ensure uniqueness
        seen = set()
        dedup = []
        for item in rec:
            if item.lower() not in seen:
                seen.add(item.lower())
                dedup.append(item)
        return dedup
    
    def _generate_ercp_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """
        Args:
            data: Extracted procedure field data
        Returns:
            List of recommendations
        """
        rec = []
        if data.get('specimens_removed', False):
            rec.append("Follow up pathology results.")
        rec.append("Finish IV fluids now.")
        rec.append("Pain control as needed.")
        
        # Biliary stent recommendations
        biliary_stent = data.get('biliary_stent_type', '').lower()
        if biliary_stent == 'plastic':
            plastic_size = data.get('plastic_biliary_size', '')
            if '5f' in str(plastic_size).lower():
                rec.append("Repeat ERCP in 6-8 weeks for stent removal/replacement.")
            # '7f', '10f', or unknown:
            else:
                rec.append("Repeat ERCP in 3-4 months for stent removal/replacement.")
        
        elif 'metal' in biliary_stent:
            metal_type = data.get('metal_biliary_type', '').lower()
            if 'fcsems' in metal_type:
                if data.get('malignancy_history', False) == True:
                    rec.append("Repeat ERCP for stent removal/replacement as clinically indicated.")
                else:
                    rec.append("Repeat ERCP in 6 months for stent removal/replacement.")
            elif 'ucsems' in metal_type:
                pass # no routine repeat ercp needed #? state?
        
        # Pancreatic stent recommendations
        if data.get('pd_stent_placed'):
            pd_purpose = data.get('pd_stent_purpose', '').lower()
            if 'pep_prophylaxis' in pd_purpose:
                rec.append("AXR in 2-4 weeks to confirm PD stent passage.")
                # If patient had indication of chronic pancreatitis/pancreatic duct strictures/pancreatic duct leaks: Repeat ERCP for stent exchange in 3 months 
            elif data.get('stent_indications_3mo', False):
                rec.append("Repeat ERCP in 3 months for stent exchange.")
            else:
                rec.append("Repeat ERCP for stent removal/replacement as clinically indicated.")
        if data.get('chol_indications', False):
            rec.append('Proceed with Cholecystectomy; can remove stent after Cholecystectomy tentatively in 3 months.')
        
        #TODO Recommendation for EDGE (EUS gastro-gastrostomy + ERCP): “Return in 4 weeks for biliary/pancreatic stent removal (if placed) and GG LAMS removal/GG fistula closure” 
        
        rec.append("Follow up with referring provider.")
        return rec
    
    
    def render(
        self,
        extracted_data: Union[BaseModel, Dict[str, Any]],
        drafter_config_path: Optional[Union[str, Path]] = None,
        format: str = 'markdown'
    ) -> Union[str, Document]:
        """Main entry point for rendering ERCP notes"""
        if isinstance(extracted_data, BaseModel):
            data_dict = extracted_data.model_dump()
        else:
            data_dict = extracted_data
        
        # Render sections from YAML templates if config provided
        rendered_sections = {}
        if drafter_config_path:
            try:
                rendered_sections = build_report_sections(str(drafter_config_path), data_dict)
            except Exception as e:
                print(f"Warning: Failed to render sections from template: {e}")
        
        if format == 'docx':
            return self.render_docx(data_dict, rendered_sections)
        else:
            return self.render_markdown(data_dict, rendered_sections)

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import sys
from pathlib import Path

_here = Path(__file__)
_root = _here.parent
sys.path.insert(0, str(_root))

from llm.client import LLMClient


class ProcedureClassification(BaseModel):
    """Result from procedure classification"""
    procedure_type: str = "ercp"
    base_template: str = "ercp/base"
    active_modules: List[str] = Field(default_factory=list)
    reasoning: str = ""

class ProcedureClassifier:
    """Classifies procedures and identifies applicable modules"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config_name: str = "openai_gpt4o"):
        """
        Args:
            llm_client: Optional pre-configured LLMClient
            config_name: Config name to use if llm_client not provided
        """
        self.llm_client = llm_client or LLMClient.from_config(config_name)
    
    def classify_procedure(self, transcript: str) -> ProcedureClassification:
        """
        Determine applicable modules and procedure type via LLM
        Args:
            transcript: Raw procedure transcript text
        Returns:
            ProcedureClassification with identified modules and type
        """
        
        ''' COMMENTED OUT FOR NOW
        2. What procedure subtype(s) were performed? Choose all that apply:
            - 1.1: Gastrogastrostomy (GG) LAMS
            - 1.2 EUS-guided GG
            - 1.3 Enteroscopy-Assisted ERCP in RYGB
            - 1.4 ERCP after Whipple or Pancreaticoduodenectomy
            - 2.1: Simple choledocholithiasis (stones <10mm, standard extraction)
            - 2.2: Complex choledocholithiasis (large stones, lithotripsy used)
            - 3.1 Gallbladder Drainage
            - 4.1 Post-cholecystectomy Bile Leak
            - 5.1 Benign Biliary Stricture
            - N/A (leave blank)
            '''
        classification_prompt = f"""Analyze this ERCP procedure transcript and identify key maneuvers and findings:
1. Which maneuvers were performed? Choose the number code(s) for all that apply:
   - 0.1: Difficult biliary cannulation
   - 0.2: Stone extraction
   - 0.3: Cholangioscopy
   - 0.4: Stent management
   - 0.5: Hemostasis
   - 0.6: Perforation/extravasation

Do this by analyzing the procedure information. Use both indications of the procedure and actual maneuvers performed.

Return your analysis in JSON format, e.g:
{{
  "active_modules": ["0.2", "0.4"],
  "reasoning": "Brief explanation of classification"
}}

Transcript:
{transcript}"""
        
        messages = [
            {"role": "system", "content": "You are an expert endoscopy procedure classifier. Return only valid JSON."},
            {"role": "user", "content": classification_prompt}
        ]
        classification_data = self.llm_client.chat_llm(messages)
        print("LLM Classification Response:", classification_data)
        active_module_codes = classification_data.get("active_modules", [])
        all_modules = ["base"] + active_module_codes
        
        return ProcedureClassification(
            procedure_type="ercp",
            base_template="ercp/base",
            active_modules=all_modules,
            reasoning=classification_data.get("reasoning", ""),
            confidence_scores=classification_data.get("confidence_scores", {})
        )
    
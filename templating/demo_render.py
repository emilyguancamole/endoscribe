from __future__ import annotations
import os
import sys

# Ensure project root is on sys.path when running this file directly
_here = os.path.dirname(__file__)
_root = os.path.normpath(os.path.join(_here, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from templating.drafter_engine import build_report_sections


def demo_scenario_1():
    """Complex case with abnormal scout film and difficult scope advancement"""
    print("=" * 80)
    print("SCENARIO 1: Complex ERCP with abnormal findings")
    print("=" * 80)
    
    here = os.path.dirname(__file__)
    cfg = os.path.normpath(os.path.join(here, "../drafters/procedures/ercp/cholangioscopy.yaml"))
    data = {
        "age": "65",
        "sex": "male",
        "chief_complaints": "obstructive jaundice",
        "symptoms_duration": "2 weeks",
        "symptoms_description": "progressive yellowing of skin and dark urine",
        "negative_history": "none",
        "past_medical_history": "hypertension and hyperlipidemia",
        "current_medications": "lisinopril, atorvastatin",
        "family_history": "mother with pancreatic cancer",
        "social_history": "nonsmoker, occasional alcohol use",
        "duodenoscope_type": "TJF-Q190V",
        "grade_of_ercp": "3",
        "pd_cannulation": "successful",
        "scout_film_status": "obtained_abnormal",
        "scout_film_findings": "surgical clips in the RUQ, pneumobilia",
        "scope_advancement_difficulty": "difficult",
        "upper_gi_examination": "limited",
        "upper_gi_findings": "erythematous gastric mucosa, periampullary diverticulum",
    }
    sections = build_report_sections(cfg, data)
    print("\n=== INDICATIONS ===")
    print(sections.get("indications", ""))
    print("\n=== HISTORY ===")
    print(sections.get("history", ""))
    print("\n=== DESCRIPTION OF PROCEDURE ===")
    print(sections.get("description_of_procedure", ""))
    print("\n=== ERCP QUALITY METRICS ===")
    print(sections.get("ercp_quality_metrics", ""))
    print("\n=== FINDINGS ===")
    print(sections.get("findings", ""))


def demo_scenario_2():
    """Simple case with normal scout film and easy scope advancement"""
    print("\n\n" + "=" * 80)
    print("SCENARIO 2: Straightforward ERCP with normal findings")
    print("=" * 80)
    
    here = os.path.dirname(__file__)
    cfg = os.path.normpath(os.path.join(here, "../drafters/procedures/ercp/config_nfu.yaml"))
    data = {
        "age": "48",
        "sex": "female",
        "chief_complaints": "biliary stent removal",
        "symptoms_duration": "none",
        "symptoms_description": "none",
        "negative_history": "fever, chills, abdominal pain",
        "past_medical_history": "cholecystectomy 3 months ago",
        "current_medications": "none",
        "family_history": "unknown",
        "social_history": "nonsmoker",
        "duodenoscope_type": "TJF-Q180V",
        "grade_of_ercp": "1",
        "pd_cannulation": "not attempted",
        "scout_film_status": "obtained_normal",
        "scout_film_findings": "none",
        "scope_advancement_difficulty": "easy",
        "upper_gi_examination": "limited",
        "upper_gi_findings": "normal",
    }
    sections = build_report_sections(cfg, data)
    print("\n=== INDICATIONS ===")
    print(sections.get("indications", ""))
    print("\n=== HISTORY ===")
    print(sections.get("history", ""))
    print("\n=== DESCRIPTION OF PROCEDURE ===")
    print(sections.get("description_of_procedure", ""))
    print("\n=== ERCP QUALITY METRICS ===")
    print(sections.get("ercp_quality_metrics", ""))
    print("\n=== FINDINGS ===")
    print(sections.get("findings", ""))


def demo():
    demo_scenario_1()
    demo_scenario_2()


if __name__ == "__main__":
    demo()

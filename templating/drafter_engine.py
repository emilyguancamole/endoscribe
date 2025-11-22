from __future__ import annotations
from typing import Dict, Any
from .config_loader import load_procedure_config
from .renderer import render_sections


def build_report_sections(config_fp: str, data: Dict[str, Any]) -> Dict[str, str]:
    """Load a procedure config and render its templates with provided data.
    Returns a map of section_name -> rendered text.
    """
    cfg = load_procedure_config(config_fp)
    templates = cfg.get("templates", {})
    return render_sections(templates, data)


__all__ = ["build_report_sections"]

from __future__ import annotations
from typing import Any, Dict
import re
from jinja2 import Environment, StrictUndefined, Template


def _filter_skip_unknown(value: Any) -> str:
    """Return empty string if a field is unknown/none-like to allow sentence omission.
    """
    return "" if _is_unknown(value) else str(value)


def _filter_sentence(text: str) -> str:
    """Normalize spacing and ensure a trailing period if not present (for non-empty)."""
    if not text:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    if t[-1] in ".!?":
        return t
    return t + "."


def _filter_capfirst(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()
    return t[0:1].upper() + t[1:]


def _is_unknown(value: Any) -> bool:
    if value is None: return True
    if isinstance(value, (int, float)):
        try:
            return float(value) == -1.0
        except Exception:
            pass
    s = str(value).strip().lower()
    if not s: return True
    if s in {"unknown", "none", "n/a", "na"}: return True
    try:
        if float(s) == -1.0:
            return True
    except Exception:
        pass

    return False


def _filter_default_if_unknown(value: Any, default: str = "") -> str:
    return default if _is_unknown(value) else str(value)


def _filter_join_nonempty(values: Any, sep: str = ", ") -> str:
    if not values:
        return ""
    parts = []
    for v in values:
        if not _is_unknown(v):
            s = str(v).strip()
            if s:
                parts.append(s)
    return sep.join(parts)


def build_env() -> Environment:
    env = Environment(undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters["skip_unknown"] = _filter_skip_unknown
    env.filters["sentence"] = _filter_sentence
    env.filters["capfirst"] = _filter_capfirst
    env.filters["default_if_unknown"] = _filter_default_if_unknown
    env.filters["join_nonempty"] = _filter_join_nonempty
    return env


def render_text(template_str: str, data: Dict[str, Any], env: Environment | None = None) -> str:
    env = env or build_env()
    tpl: Template = env.from_string(template_str)
    # print("data:", data)  # Debug: uncomment to see data dict
    rendered = tpl.render(**data)
    if not rendered:
        return ""
    # Normalize blank lines: collapse runs of blank/whitespace-only lines to a single blank line and trim leading/trailing whitespace/newlines.
    # Convert Windows CRLF to LF first for safety.
    text = rendered.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def render_sections(sections: Dict[str, str], data: Dict[str, Any], env: Environment | None = None) -> Dict[str, str]:
    env = env or build_env()
    out: Dict[str, str] = {}
    for key, tmpl in sections.items():
        out[key] = render_text(tmpl, data, env)
    return out


__all__ = ["build_env", "render_text", "render_sections"]

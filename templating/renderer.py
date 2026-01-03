from __future__ import annotations
from typing import Any, Dict
import re
from jinja2 import Environment, ChainableUndefined, Template
from jinja2.runtime import Undefined
import datetime


def _filter_skip_unknown(value: Any) -> str:
    """Return empty string if a field is unknown/none-like to allow sentence omission.
    """
    if isinstance(value, Undefined):
        return ""
    return "" if _is_unknown(value) else str(value)


def _filter_sentence(text: str) -> str:
    """Normalize spacing, capitalize first word (unless begins with a number), and ensure a trailing sentence terminator
    """
    if not text:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    first_token = t.split(None, 1)[0] if t.split(None, 1) else t
    if not first_token[0].isdigit():
        for i, ch in enumerate(t):
            if ch.isalpha():
                t = t[:i] + ch.upper() + t[i+1:]
                break

    if t[-1] in '.!?':
        return t
    return t + '.'


def _filter_period(text: str) -> str:
    """Add a period to the end of text only if it doesn't already end with . ! ?.
    """
    if not text:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    if t[-1] in '.!?':
        return t
    return t + '.'


def _filter_capfirst(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()
    return t[0:1].upper() + t[1:]


def _is_unknown(value: Any) -> bool:
    if isinstance(value, Undefined):
        return True
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
    if isinstance(value, Undefined):
        return default
    return default if _is_unknown(value) else str(value)


def _filter_default_if_unknown_sentence(value: Any, default: str = "") -> str:
    """Return the `default` if value is unknown; otherwise return the value.
    Ensure output is capitalized and ends with a sentence terminator.
    Useful for short free-text sections where you want a nicely formatted fallback.
    """
    s = _filter_default_if_unknown(value, default)
    if not s:
        return ""
    return _filter_sentence(_filter_capfirst(s))


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


def _filter_today(value: Any = None, fmt: str = "%Y-%m-%d") -> str:
    """Return today's date formatted by `fmt`.
    - as a global: `{{ today("%b %d, %Y") }}`
    - as a filter: `{{ "" | today("%b %d, %Y") }}`
    """
    try:
        return datetime.date.today().strftime(fmt)
    except Exception:
        return datetime.date.today().isoformat()


def build_env() -> Environment:
    ''' Register Jinja environment with custom filters'''
    env = Environment(undefined=ChainableUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters["sku"] = _filter_skip_unknown
    env.filters["sent"] = _filter_sentence
    env.filters["capfirst"] = _filter_capfirst
    env.filters["default_if_unknown"] = _filter_default_if_unknown
    env.filters["default_if_unknown_sentence"] = _filter_default_if_unknown_sentence
    env.filters["period"] = _filter_period
    env.filters["join_nonempty"] = _filter_join_nonempty
    env.filters["today"] = _filter_today
    env.globals["today"] = lambda fmt: _filter_today(None, fmt)
    return env


def render_text(template_str: str, data: Dict[str, Any], env: Environment | None = None) -> str:
    env = env or build_env()
    tpl: Template = env.from_string(template_str)
    rendered = tpl.render(**data)
    if not rendered:
        return ""
    # Normalize blank lines: collapse runs of blank/whitespace-only lines to a single blank line and trim leading/trailing whitespace/newlines.
    text = rendered.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def render_sections(sections: Dict[str, str], data: Dict[str, Any], env: Environment | None = None) -> Dict[str, str]:
    env = env or build_env()
    out: Dict[str, str] = {}
    for key, tmpl in sections.items():
        try:
            out[key] = render_text(tmpl, data, env)
        except Exception as e:
            # Debug Jinja syntax errors
            snippet = tmpl[:1000].replace('\n', '\\n') if tmpl else ''
            raise RuntimeError(f"Error rendering section '{key}': {e}\nTemplate snippet: {snippet}") from e
    return out


__all__ = ["build_env", "render_text", "render_sections"]

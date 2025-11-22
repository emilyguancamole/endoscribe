import os
from copy import deepcopy
from typing import Any, Dict
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. Child values override base; nested dicts merged recursively.
    Lists are replaced by default.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_yaml(fp: str) -> Dict[str, Any]:
    with open(fp, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_path(base_path: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(os.path.dirname(base_path), rel_or_abs))


def load_procedure_config(config_path: str) -> Dict[str, Any]:
    """Load a procedure config that may extend another via "$extends": "path/to/base.json".
    Child overrides are deep-merged over base. Paths can be relative to the current file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cur = load_yaml(config_path)
    base_cfg = {}
    if "$extends" in cur and cur["$extends"]:
        base_path = resolve_path(config_path, cur["$extends"])
        base_cfg = load_procedure_config(base_path)

    # Remove the $extends directive from final dict
    cur_no_ext = {k: v for k, v in cur.items() if k != "$extends"}
    return deep_merge(base_cfg, cur_no_ext)


__all__ = ["deep_merge", "load_procedure_config"]

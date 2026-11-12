from __future__ import annotations

from .model import read_json

MODES = ("full", "select", "no_prune", "baseline", "drain", "disabled")
BYPASS_MODES = ("drain", "disabled")
BASELINE_MODES = ("baseline", *BYPASS_MODES)


def configured_mode(path: str | None, default: str = "full") -> str:
    data = read_json(path) if path else {}
    if not isinstance(data, dict):
        raise ValueError("FastCI configuration must be a JSON object")
    mode = data.get("mode", default)
    if mode not in MODES:
        raise ValueError("Unknown FastCI configuration mode")
    return mode

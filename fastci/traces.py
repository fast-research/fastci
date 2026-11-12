from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

from .graph import Evidence
from .model import read_json
from .source import SourceIndex, entity

_SECONDS_PER_UNIT = {
    "seconds": 1,
    "milliseconds": 0.001,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}


def _frame_name(frame, root, index):
    if not isinstance(frame.get("name"), str):
        raise ValueError("Trace frame names must be strings")
    file_path = frame.get("file")
    if not isinstance(file_path, str):
        return None
    try:
        relative = PurePosixPath(file_path).relative_to(root)
    except ValueError:
        return None
    if ".." in relative.parts or not str(relative).endswith(".py"):
        return None
    file = str(relative)
    raw = frame["name"].replace(".<locals>.", ".")
    function = index.at_line(file, int(frame.get("line") or 0))
    # A callsite line may point to an enclosing scope. Check the leaf name before qualifying.
    if function and function.name.rsplit(".", 1)[-1] == raw.rsplit(".", 1)[-1]:
        return function.key
    return entity(file, raw)


def _sampled_profiles(data, path):
    if not isinstance(data, dict) or not isinstance(data.get("shared"), dict):
        raise ValueError("Trace must contain shared frame data")
    frames = data["shared"]["frames"]
    if not isinstance(frames, list) or not all(
        isinstance(frame, dict) for frame in frames
    ):
        raise ValueError("Trace frames must be objects")
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not all(
        isinstance(profile, dict) for profile in profiles
    ):
        raise ValueError("Trace profiles must be objects")
    sampled = [profile for profile in profiles if profile.get("type") == "sampled"]
    if not sampled:
        raise ValueError(f"Not a sampled Speedscope trace: {path}")
    return frames, sampled


def _merge_samples(evidence, profile, names):
    factor = _SECONDS_PER_UNIT.get(profile.get("unit"))
    if factor is not None:
        duration = (profile["endValue"] - profile["startValue"]) * factor
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid trace duration")
        # Sampled threads overlap, so retain the longest observed duration.
        if evidence.seconds is None or duration > evidence.seconds:
            evidence.seconds = duration
    if not isinstance(profile.get("samples"), list):
        raise ValueError("Sampled profiles must contain a list of stacks")
    for stack in profile["samples"]:
        if not isinstance(stack, list):
            raise ValueError("Sample stacks must be lists of frame indices")
        if any(type(i) is not int or i < 0 or i >= len(names) for i in stack):
            raise ValueError("Invalid Speedscope frame index")
        retained = [names[i] for i in stack if names[i] is not None]
        evidence.nodes.update(retained)
        evidence.edges.update(zip(retained, retained[1:]))


def parse_speedscope(
    paths: list[str | Path], remote_root: str, index: SourceIndex
) -> Evidence:
    """Combine every sampled profile and reject malformed evidence."""
    evidence = Evidence()
    root = PurePosixPath(remote_root)
    for path in paths:
        frames, profiles = _sampled_profiles(read_json(path), path)
        names = [_frame_name(frame, root, index) for frame in frames]
        for profile in profiles:
            _merge_samples(evidence, profile, names)
    if not evidence.nodes:
        raise ValueError(
            "No usable repository evidence; preserve the previous fragment"
        )
    return evidence

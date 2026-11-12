from __future__ import annotations

import fnmatch
from dataclasses import asdict, dataclass, field
from typing import Any

from .graph import Graph
from .model import TestSpec, digest
from .risk import OBSERVED_OUTCOMES


@dataclass
class Proposal:
    overrides: dict[str, Any] = field(default_factory=dict)
    blocked: dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""


def propose(test: TestSpec, graph: Graph, rules: dict) -> Proposal:
    """Reduction requires explicit validation-scope evidence, never a name guess."""
    overrides, blocked = {}, {}
    nodes = graph.tests[test.name].nodes if test.name in graph.tests else set()
    protected = set(test.metadata.get("validation_dimensions", []))
    unrelated = set(test.metadata.get("unrelated_dimensions", []))
    paths = rules.get("path_dimensions", {})
    for pattern, dimensions in paths.items():
        if any(fnmatch.fnmatchcase(node, pattern) for node in nodes):
            protected.update(dimensions)
    for dimension, value in rules.get("reductions", {}).items():
        if dimension not in test.workload or test.workload[dimension] == value:
            continue
        if dimension in protected:
            blocked[dimension] = "validation_scope_or_observed_path"
        elif dimension not in unrelated:
            blocked[dimension] = "scope_unknown"
        elif not test.bindings.get(dimension, {}).get("key") or test.bindings[
            dimension
        ].get("kind") not in ("env", "arg"):
            blocked[dimension] = "no_runner_binding"
        else:
            overrides[dimension] = value
    fingerprint = digest([asdict(test), rules, overrides])
    return Proposal(overrides, blocked, fingerprint)


@dataclass
class WorkloadGuard:
    entries: dict[str, dict] = field(default_factory=dict)
    interval: int = 10

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("Paired validation interval must be positive")

    def key(self, name: str, proposal: Proposal, environment: str) -> str:
        return digest([name, proposal.fingerprint, environment])

    def action(self, name: str, proposal: Proposal, environment: str) -> str:
        if not proposal.overrides:
            return "original"
        state = self.entries.get(self.key(name, proposal, environment), {})
        if state.get("revoked"):
            return "original"
        if not state.get("accepted") or state.get("uses", 0) >= self.interval - 1:
            return "paired"
        return "reduced"

    def compare(
        self,
        name: str,
        proposal: Proposal,
        environment: str,
        original: str,
        reduced: str,
        revision: str,
    ) -> bool:
        agree = original == reduced and original in OBSERVED_OUTCOMES
        self.entries[self.key(name, proposal, environment)] = {
            "accepted": agree,
            "revoked": not agree,
            "uses": 0,
            "revision": revision,
            "original": original,
            "reduced": reduced,
        }
        return agree

    def used(self, name: str, proposal: Proposal, environment: str) -> None:
        self.entries[self.key(name, proposal, environment)]["uses"] += 1


def invocation(
    test: TestSpec, overrides: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    command, environment = list(test.command), dict(test.env)
    for dimension, value in overrides.items():
        binding = test.bindings[dimension]
        key = binding["key"]
        text = str(value).lower() if isinstance(value, bool) else str(value)
        if binding["kind"] == "env":
            if key == "CUDA_VISIBLE_DEVICES":
                raise ValueError("A workload override cannot change GPU allocation")
            environment[key] = text
        else:
            _replace_argument(command, key, text, binding.get("style"))
    return command, environment


def _replace_argument(
    command: list[str], key: str, value: str, style: str | None
) -> None:
    """Update every spelling of a flag and preserve argv boundaries."""
    equals = key + "="
    matched, position = False, 0
    while position < len(command):
        if command[position] == key:
            if position + 1 == len(command):
                raise ValueError(f"Missing argument value for {key}")
            command[position + 1] = value
            matched = True
            position += 2
            continue
        if command[position].startswith(equals):
            command[position] = equals + value
            matched = True
        position += 1
    if not matched:
        command.extend((equals + value,) if style == "equals" else (key, value))

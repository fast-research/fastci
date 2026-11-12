from __future__ import annotations

from dataclasses import asdict

from .graph import Graph
from .model import TestSpec, suite_json
from .policy import BASELINE_MODES, MODES
from .risk import RiskHistory
from .selection import Selection, select
from .source import Change, SourceIndex
from .workload import propose


def plan(
    change: Change,
    graph: Graph,
    suite: dict[str, TestSpec],
    indexes: tuple[SourceIndex, ...],
    history: RiskHistory,
    rules: dict,
    baseline: list[str] | None = None,
    mode: str = "full",
    depth: int = 3,
    *,
    file_jobs: dict[str, set[str]] | None = None,
) -> dict:
    """Return a reviewable plan; planning never advances history or guard state."""
    baseline = list(suite) if baseline is None else baseline
    if len(set(baseline)) != len(baseline) or set(baseline) - suite.keys():
        raise ValueError("Baseline must contain unique registered tests")
    if mode not in MODES:
        raise ValueError("Unknown mode")
    if mode in BASELINE_MODES:
        selection = Selection(baseline, baseline, {}, fallbacks=["existing_ci_rules"])
    else:
        selection = select(
            change,
            graph,
            suite,
            indexes,
            baseline,
            depth,
            mode != "no_prune",
            file_jobs=file_jobs,
        )
    affected, retained = selection.affected, selection.retained
    order, scores = (
        history.order(retained, change.files) if mode == "full" else (retained, {})
    )
    proposals = {
        name: asdict(propose(suite[name], graph, rules if mode == "full" else {}))
        for name in order
    }
    return {
        "schema": 1,
        "mode": mode,
        "change": asdict(change),
        "depth": depth,
        "affected": affected,
        "retained": retained,
        "order": order,
        "baseline": baseline,
        "scores": scores,
        "proposals": proposals,
        "selection": asdict(selection),
        "suite": suite_json(suite),
        "graph_revision": graph.revision,
    }

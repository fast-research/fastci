from __future__ import annotations

from dataclasses import dataclass, field

from .graph import Graph
from .model import TestSpec
from .source import Change, SourceIndex


@dataclass
class Selection:
    affected: list[str]
    retained: list[str]
    reasons: dict[str, list[str]]
    groups: list[dict] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)


def _context_groups(modified, candidates, graph, suite, depth):
    equivalence: dict[tuple, list[str]] = {}
    for name in suite:  # deterministic original suite order
        if name in candidates:
            key = (graph.context(modified, name, depth), suite[name].validation_key())
            equivalence.setdefault(key, []).append(name)
    for members in equivalence.values():
        representative = min(
            members,
            key=lambda n: suite[n].cost(
                graph.tests[n].seconds,
                measured=graph.tests[n].timing_source == "execution",
            ),
        )
        yield {"entity": modified, "members": members, "representative": representative}


def select(
    change: Change,
    graph: Graph,
    suite: dict[str, TestSpec],
    indexes: tuple[SourceIndex, ...],
    baseline: list[str],
    depth: int = 3,
    prune: bool = True,
    *,
    file_jobs: dict[str, set[str]] | None = None,
) -> Selection:
    if depth < 0:
        raise ValueError("Context depth must be nonnegative")
    if set(baseline) - suite.keys():
        raise ValueError("Baseline refers to tests missing from the suite")
    selected: set[str] = set()
    retained: set[str] = set()
    reasons: dict[str, list[str]] = {}
    fallbacks = list(change.unresolved)
    groups = []
    full_fallback = False
    for problem in change.unresolved:
        path = problem.rsplit(":", 1)[0]
        candidates = (
            file_jobs.get(path, set()) & suite.keys()
            if file_jobs is not None
            else set()
        )
        if candidates:
            selected.update(candidates)
            retained.update(candidates)
            reasons[path] = sorted(candidates)
            fallbacks.append(path + ":file_evidence")
        else:
            full_fallback = True
    # Non-Python changes are delegated to caller-supplied existing CI rules.
    if any(not file.endswith(".py") for file in change.files):
        selected.update(baseline)
        retained.update(baseline)
        fallbacks.append("non_python:existing_ci_rules")
    untraced = set(suite) - {name for name, e in graph.tests.items() if e.nodes}
    if change.files and untraced:
        selected.update(untraced)
        retained.update(untraced)
        fallbacks.append("untraced_tests:retain")
    coverage = graph.coverage_by_node()
    for modified in change.entities:
        direct = coverage.get(modified, set()) & suite.keys()
        if direct:
            candidates = direct
            expanded = False
        else:
            expanded_entities = set().union(*(idx.expand(modified) for idx in indexes))
            candidates = (
                set().union(*(coverage.get(entity, ()) for entity in expanded_entities))
                & suite.keys()
            )
            expanded = True
            if not candidates and file_jobs is not None:
                paths = {
                    idx.functions[modified].path
                    for idx in indexes
                    if modified in idx.functions
                }
                candidates = (
                    set().union(*(file_jobs.get(path, set()) for path in paths))
                    & suite.keys()
                )
                if candidates:
                    fallbacks.append(modified + ":file_evidence")
            if not candidates:
                fallbacks.append(modified + ":no_expansion_evidence")
                # File-scoped selection reports a miss and keeps the union of
                # available evidence. The adapter guards an entirely empty plan.
                full_fallback |= file_jobs is None
        selected.update(candidates)
        reasons[modified] = sorted(candidates)
        # Missing original context cannot establish equivalence for modified m.
        if expanded or not prune:
            retained.update(candidates)
            continue
        for group in _context_groups(modified, candidates, graph, suite, depth):
            groups.append(group)
            retained.add(group["representative"])
    if file_jobs is not None and change.files and not any(reasons.values()):
        fallbacks.append("no_scoped_evidence")
        full_fallback = True
    if full_fallback:
        selected.update(baseline)
        retained.update(baseline)
    return Selection(
        [n for n in suite if n in selected],
        [n for n in suite if n in retained],
        reasons,
        groups,
        fallbacks,
    )

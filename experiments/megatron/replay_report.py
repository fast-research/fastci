"""Score the fixed coverage oracle and format the Megatron benchmark report."""

from __future__ import annotations

from collections import defaultdict
from itertools import repeat
from math import fsum, isclose
from statistics import mean

FASTCI_STRATEGIES = {
    "no_cep_no_file": "FastCI without CEP or file matching",
    "no_cep_file": "FastCI without CEP, with file matching",
    "cep_no_file": "FastCI with CEP, without file matching",
    "cep_file": "FastCI with CEP and file matching",
}
STRATEGIES = {
    "full": "Full CI",
    "path": "Megatron L0",
    "llm": "Codex Planner",
    "namerts": "NameRTS (Megatron)",
    **FASTCI_STRATEGIES,
}


def _score(commit, selections, costs):
    affected = set(commit["affected"])
    regions = {
        name: {
            (file_index, line) for file_index, lines in hits.items() for line in lines
        }
        for name, hits in commit["line_hits"].items()
    }
    full = set().union(*regions.values())
    results = {}
    for method, selected in selections.items():
        chosen = set(selected)
        if len(chosen) != len(selected) or chosen - costs.keys():
            raise ValueError(f"Invalid selections for {method} on {commit['commit']}")
        covered = set().union(*(regions.get(name, set()) for name in chosen))
        results[method] = {
            "selected": selected,
            "recall": len(chosen & affected) / len(affected) if affected else None,
            "retention": len(covered) / len(full) if affected and full else None,
            "execution_gpu_seconds": fsum(costs[name] for name in chosen),
        }
    return results


def _summarize(rows, methods, graph_seconds, baseline_seconds):
    affected = [row for row in rows if row["affected"]]
    return {
        "commits": len(rows),
        "affected_commits": len(affected),
        "full_ci_gpu_seconds": baseline_seconds * len(rows),
        "methods": {
            method: _summarize_method(
                rows, affected, method, graph_seconds, baseline_seconds
            )
            for method in methods
        },
    }


def _summarize_method(rows, affected, method, graph_seconds, baseline_seconds):
    full = baseline_seconds * len(rows)
    execution = fsum(row["scores"][method]["execution_gpu_seconds"] for row in rows)
    build = graph_seconds if method in FASTCI_STRATEGIES else 0
    recall = [row["scores"][method]["recall"] for row in affected]
    retention = [row["scores"][method]["retention"] for row in affected]
    return {
        "recall": mean(recall) if recall else None,
        "retention": mean(retention) if retention else None,
        "execution_gpu_seconds": execution,
        "graph_gpu_seconds": build,
        "net_gpu_seconds": execution + build,
        "net_gpu_savings": 1 - (execution + build) / full if full else None,
    }


def build_report(data, commits, verified_selections=None):
    costs = data["unsampled_gpu_seconds"]
    graph_seconds = fsum(data["graph_build_attempt_gpu_seconds"])
    baseline_seconds = fsum(costs.values())
    if verified_selections is None:
        verified_selections = repeat(({}, {}), len(commits))
    rows = []
    cohorts = defaultdict(list)
    for commit, (current, diagnostics) in zip(
        commits, verified_selections, strict=True
    ):
        decisions = {
            "full": list(costs),
            "path": data["l0"],
            **commit["selections"],
            **current,
        }
        row = {
            "commit": commit["commit"],
            "parent": commit["parent"],
            "cohort": commit["cohort"],
            "upstream_pr": commit["upstream_pr"],
            "affected": commit["affected"],
            "scores": _score(commit, decisions, costs),
            **diagnostics,
        }
        rows.append(row)
        cohorts[row["cohort"]].append(row)
    methods = list(rows[0]["scores"])
    groups = {
        name: _summarize(group, methods, graph_seconds, baseline_seconds)
        for name, group in (("all", rows), *sorted(cohorts.items()))
    }
    return {"reference": data["reference"], "groups": groups, "results": rows}


def format_table(report):
    methods = report["groups"]["all"]["methods"]
    lines = ["strategy\trecall_percent\tretention_percent\tnet_gpu_savings_percent"]
    for method, label in STRATEGIES.items():
        entry = methods[method]
        saving = entry["net_gpu_savings"]
        lines.append(
            f"{label}\t{100 * entry['recall']:.2f}\t{100 * entry['retention']:.2f}\t"
            f"{100 * (0 if isclose(saving, 0, abs_tol=1e-12) else saving):.2f}"
        )
    return "\n".join(lines) + "\n"

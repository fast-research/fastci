from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..graph import Graph
from ..model import read_json
from ..source import SourceIndex, sources_at
from ..traces import parse_speedscope
from .common import RANKS, file_hash, seal, trace_phases, verify, within


def _verified_traces(outcome, run_root, repeats, phases):
    if outcome["status"] != "passed" or outcome["trace_errors"]:
        raise ValueError("failed test or incomplete sampling")
    expected = {
        (repeat, phase, rank)
        for repeat, phase in trace_phases(repeats, phases)
        for rank in RANKS
    }
    traces = outcome["traces"]
    if (
        len(traces) != len(expected)
        or {(t["repeat"], t["phase"], t["rank"]) for t in traces} != expected
    ):
        raise ValueError("missing repeat, phase, or rank traces")
    root = within(outcome["source_root"], run_root)
    paths = []
    for trace in traces:
        path = within(run_root / trace["path"], root)
        if path in paths or file_hash(path) != trace["sha256"]:
            raise ValueError("duplicate trace or trace checksum mismatch")
        paths.append(path)
    return paths, root


def ingest(run_path):
    """Publish only a complete fresh graph for the run's exact source revision."""
    run_path = Path(run_path).resolve()
    result = read_json(run_path)
    verify(result)
    plan = result["plan"]
    verify(plan)
    catalog = plan["catalog"]
    verify(catalog)
    if result["kind"] != "native_gpu_run" or not result["runtime"]:
        raise ValueError("A native GPU run is required for graph ingestion")
    if result["revision"] != catalog["revision"]:
        raise ValueError("Run revision differs from its catalog")
    if [o["test"] for o in result["outcomes"]] != plan["execute"]:
        raise ValueError("Run outcomes do not match the plan")
    cfg = catalog["config"]
    rows = {row["id"]: row for row in catalog["tests"]}
    graph = Graph(revision=result["revision"])
    index = SourceIndex(sources_at(result["source_repo"], result["revision"]))
    errors = {}
    for outcome in result["outcomes"]:
        name = outcome["test"]
        try:
            paths, root = _verified_traces(
                outcome,
                run_path.parent,
                cfg["n_repeat"],
                rows[name].get("training_phases", 1),
            )
            fragment = parse_speedscope(paths, str(root), index)
            graph.tests[name] = replace(
                fragment, seconds=outcome["seconds"], timing_source="execution"
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors[name] = str(error)
    return seal(
        {
            "schema": 1,
            "revision": result["revision"],
            "graph": graph.to_json(),
            "environment_id": cfg["environment_id"],
            "hardware": cfg["hardware"],
            "runtime_fingerprint": result["runtime"]["fingerprint"],
            "complete_baseline": graph.tests.keys() == rows.keys(),
            "fingerprints": {name: rows[name]["fingerprint"] for name in graph.tests},
            "errors": errors,
            "run_digest": result["digest"],
        }
    )


def evaluate(result):
    verify(result)
    plan = result["plan"]
    proposed = set(plan["proposed"])
    omitted = [row for row in result["outcomes"] if row["test"] not in proposed]
    shadow = result["kind"] == "native_gpu_run" and plan["mode"] == "shadow"
    return {
        "kind": result["kind"],
        "mode": plan["mode"],
        "baseline_tests": len(plan["baseline"]),
        "proposed_tests": len(proposed),
        "fallback_reasons": plan["fallback_reasons"],
        "observed_gpu_seconds": result["gpu_seconds"],
        "prefix_outcomes": {
            row["test"]: row["prefix"] for row in result["outcomes"] if "prefix" in row
        },
        "omitted_failures": [
            row["test"] for row in omitted if row["status"] != "passed"
        ]
        if shadow
        else None,
        "counterfactual_omitted_gpu_seconds": sum(row["gpu_seconds"] for row in omitted)
        if shadow
        else None,
    }


def observe(result, history):
    verify(result)
    if result["kind"] != "native_gpu_run":
        raise ValueError("A native GPU run is required to update risk history")
    statuses = {"passed": "pass", "failed": "fail"}
    history.observe(
        result["digest"],
        result["plan"]["change"]["files"],
        {
            row["test"]: statuses[row["status"]]
            for row in result["outcomes"]
            if row["status"] in statuses
        },
    )
    return history

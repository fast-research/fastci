from __future__ import annotations

from subprocess import CalledProcessError

from ..engine import plan as core_plan
from ..graph import Graph
from ..model import TestSpec
from ..risk import RiskHistory
from ..source import analyze, jobs_by_file
from .common import GPUS_PER_NODE, MODES, git, seal, verify


def _compatible_graph(repo, evidence, cfg, base):
    if evidence is None:
        raise ValueError("missing_graph")
    verify(evidence)
    if evidence.get("complete_baseline") is not True:
        raise ValueError("incomplete_baseline")
    if (evidence["hardware"], evidence["environment_id"]) != (
        cfg["hardware"],
        cfg["environment_id"],
    ):
        raise ValueError("incompatible_environment")
    if not evidence.get("runtime_fingerprint"):
        raise ValueError("missing_runtime_fingerprint")
    if git(repo, "rev-parse", evidence["revision"] + "^{tree}") != git(
        repo, "rev-parse", base + "^{tree}"
    ):
        raise ValueError("graph_tree_differs_from_merge_base")
    graph = Graph.from_json(evidence["graph"])
    if graph.revision != evidence["revision"]:
        raise ValueError("graph_revision_mismatch")
    return graph


def plan(repo, catalog, target, evidence=None, mode="shadow", history=None):
    verify(catalog)
    if mode not in MODES:
        raise ValueError("Unknown Megatron mode")
    history = history if history is not None else RiskHistory()
    change, old, new = analyze(repo, catalog["revision"], target)
    cfg = catalog["config"]
    suite = {
        row["id"]: TestSpec(
            row["id"],
            gpus=GPUS_PER_NODE,
            hardware=cfg["hardware"],
            metadata={
                "task_type": "megatron-pretraining",
                "validation": row["validation_key"],
            },
        )
        for row in catalog["tests"]
    }
    baseline = list(suite)
    reasons = []
    graph = Graph()
    protected = set()
    if mode == "baseline":
        reasons.append("baseline_mode")
    else:
        try:
            graph = _compatible_graph(repo, evidence, cfg, change.merge_base)
            allowlist = set(cfg.get("allowlist", cfg["cases"]))
            for row in catalog["tests"]:
                name = row["id"]
                if evidence["fingerprints"].get(name) != row["fingerprint"]:
                    # A changed validation recipe cannot represent another test.
                    graph.tests.pop(name, None)
                    protected.add(name)
                if row["case"] not in allowlist:
                    protected.add(name)
        except (ValueError, KeyError, TypeError, OSError, CalledProcessError) as error:
            reasons.append(str(error))
            graph = Graph()
    file_jobs = jobs_by_file(
        old.sources.keys() | new.sources.keys(), graph.coverage_by_node()
    )
    decision = core_plan(
        change,
        graph,
        suite,
        (old, new),
        history,
        {},
        baseline,
        mode="baseline" if reasons else "full",
        file_jobs=file_jobs,
    )
    if not reasons:
        reasons.extend(decision["selection"]["fallbacks"])
    retained = set(decision["retained"]) | protected
    if not retained:
        reasons.append("empty_selection")
        retained = set(baseline)
    proposed = [name for name in baseline if name in retained]
    if mode != "baseline":
        proposed, _ = history.order(proposed, change.files)
    return seal(
        {
            "schema": 1,
            "mode": mode,
            "catalog": catalog,
            "change": decision["change"],
            "baseline": baseline,
            "proposed": proposed,
            "execute": proposed if mode == "enforce" else baseline,
            "fallback_reasons": reasons,
            "selection": decision["selection"],
            "base_evidence": evidence if graph.revision is not None else None,
            "workload_reduction": mode == "enforce"
            and not reasons
            and not protected
            and any(
                row.get("reduction") and row["id"] in retained
                for row in catalog["tests"]
            ),
        }
    )

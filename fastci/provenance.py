from __future__ import annotations

import math
from pathlib import Path

from .model import digest, file_hash
from .policy import BYPASS_MODES
from .repository import git
from .risk import OBSERVED_OUTCOMES


def verify_checkout(repo: str | Path, revision: str) -> None:
    if (
        git(repo, "rev-parse", "HEAD").strip() != revision
        or git(repo, "status", "--porcelain", "--untracked-files=no").strip()
    ):
        raise ValueError(
            "Run requires an unchanged clean checkout at the planned source revision"
        )


def validate_ingestion(
    plan: dict,
    run: dict,
    snapshot: dict,
    revision: str,
    manifest: dict,
    manifest_root: Path,
) -> tuple[dict, dict]:
    if not isinstance(manifest, dict):
        raise ValueError("Trace manifest must be a test-to-files mapping")
    if revision != plan["change"]["revision"] or revision != run.get("revision"):
        raise ValueError("Plan, run, and ingestion revisions differ")
    if run.get("plan_digest") != digest(plan) or not run.get("source_verified"):
        raise ValueError("Run is not verified against this plan and source revision")
    if run.get("effective_mode") in BYPASS_MODES:
        raise ValueError("Bypassed CI runs cannot update FastCI graphs")
    if plan["base_snapshot"]["digest"] != digest(snapshot):
        raise ValueError("Base snapshot differs from the evidence used for selection")
    outcomes = {outcome["test"]: outcome for outcome in run["outcomes"]}
    if len(outcomes) != len(run["outcomes"]):
        raise ValueError("Duplicate test outcomes")
    if set(outcomes) != set(plan["order"]):
        raise ValueError("Run test set differs from the execution plan")
    accepted, errors = {}, {}
    for test, paths in manifest.items():
        outcome = outcomes.get(test)
        if (
            outcome is None
            or not outcome.get("attempts")
            or outcome.get("status") not in OBSERVED_OUTCOMES
        ):
            errors[test] = "No execution for this test in the verified run"
            continue
        captures = {
            str(Path(item["path"]).resolve()): item["sha256"]
            for attempt in outcome["attempts"]
            for item in attempt.get("trace_files", [])
        }
        try:
            if not isinstance(paths, list) or not paths:
                raise ValueError("Trace paths must be a nonempty list")
            resolved = [manifest_root / path for path in paths]
            identities = [str(path.resolve()) for path in resolved]
            if (
                len(set(identities)) != len(resolved)
                or set(identities) != captures.keys()
            ):
                raise ValueError(
                    "Manifest must include every captured trace exactly once"
                )
            for path, identity in zip(resolved, identities):
                if file_hash(path) != captures[identity]:
                    raise ValueError("Trace does not match a captured file in this run")
            accepted[test] = resolved
        except (OSError, TypeError, ValueError) as exc:
            errors[test] = str(exc)
    return accepted, errors


def execution_seconds(outcome: dict) -> float:
    attempts = outcome["attempts"]
    authoritative = (
        attempts[0] if outcome.get("comparison_agreed") is False else attempts[-1]
    )
    seconds = authoritative["seconds"]
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid execution duration")
    return seconds

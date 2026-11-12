from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .model import TestSpec
from .risk import OBSERVED_OUTCOMES


def prepare_history(
    csv_path: str | Path, suite: dict[str, TestSpec]
) -> tuple[list[dict], dict]:
    """Convert portable CI observations to replay records.

    Required columns: run_id, sequence, test, status, files (a JSON list).
    Optional column: seconds. Sequence orders runs, not individual retries.
    Within a run, row order defines the original test order. The first terminal
    pass/fail observation of each test wins; absent outcomes are never passes.
    """
    runs: dict[str, dict] = {}
    diagnostics = {
        "rows": 0,
        "outside_suite": 0,
        "unobserved_outcomes": 0,
        "duplicate_test_observations": 0,
    }
    required = {"run_id", "sequence", "test", "status", "files"}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "History CSV requires columns: " + ", ".join(sorted(required))
            )
        for line, row in enumerate(reader, start=2):
            diagnostics["rows"] += 1
            name = row["test"]
            if name not in suite:
                diagnostics["outside_suite"] += 1
                continue
            if row["status"] in ("skipped", "timeout", "error", "cancelled"):
                diagnostics["unobserved_outcomes"] += 1
                continue
            try:
                if row["status"] not in OBSERVED_OUTCOMES:
                    raise ValueError(
                        "status must be pass, fail, skipped, timeout, error, or cancelled"
                    )
                run_id = row["run_id"]
                if not run_id:
                    raise ValueError("empty run_id")
                sequence = int(row["sequence"])
                files = json.loads(row["files"])
                if not isinstance(files, list) or not all(
                    isinstance(f, str) for f in files
                ):
                    raise ValueError("files must be a JSON array of paths")
                files = sorted(set(files))
                raw_seconds = row.get("seconds") or ""
                seconds = float(raw_seconds) if raw_seconds.strip() else None
                if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
                    raise ValueError("seconds must be finite and nonnegative")
                record = runs.setdefault(
                    run_id,
                    {
                        "run_id": run_id,
                        "sequence": sequence,
                        "files": files,
                        "jobs": {},
                    },
                )
                if record["sequence"] != sequence or record["files"] != files:
                    raise ValueError("sequence and files must agree within a run")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"History CSV line {line}: {exc}") from exc
            if name in record["jobs"]:
                diagnostics["duplicate_test_observations"] += 1
                continue
            record["jobs"][name] = {"status": row["status"], "seconds": seconds}
    sequences = [record["sequence"] for record in runs.values()]
    if len(set(sequences)) != len(sequences):
        raise ValueError("Different runs must have unique sequence values")
    records = []
    for record in sorted(runs.values(), key=lambda r: r["sequence"]):
        jobs = record["jobs"]
        records.append(
            {
                "run_id": record["run_id"],
                "sequence": record["sequence"],
                "files": record["files"],
                "tests": list(jobs),
                "outcomes": {n: job["status"] for n, job in jobs.items()},
                "seconds": {
                    n: job["seconds"]
                    for n, job in jobs.items()
                    if job["seconds"] is not None
                },
                "gpus": {
                    n: suite[n].gpus
                    for n in jobs
                    if not suite[n].metadata.get("resources_unresolved")
                },
            }
        )
    diagnostics.update(
        runs=len(records),
        all_pass_runs=sum(
            all(s == "pass" for s in r["outcomes"].values()) for r in records
        ),
    )
    return records, diagnostics

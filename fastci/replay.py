from __future__ import annotations

from .risk import OBSERVED_OUTCOMES, RiskHistory


def scheduling_replay(
    records: list[dict], warmup: int = 30, resource_mode: str = "local"
) -> dict:
    """Frozen selected sets; score before observing each chronologically sorted MR.

    This reports ordering and SERIAL PREFIX work, not actual cluster TTFF or
    cancellation savings. Full baseline outcomes are required for every selected
    test; absent results are never fabricated as passes.
    """
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if resource_mode not in ("local", "mock"):
        raise ValueError("Resource mode must be local or mock")
    cost_key = (
        "serial_prefix_mock_device_seconds"
        if resource_mode == "mock"
        else "serial_prefix_gpu_seconds"
    )
    if len({r["run_id"] for r in records}) != len(records):
        raise ValueError("Duplicate MR/run IDs")
    if len({r["sequence"] for r in records}) != len(records):
        raise ValueError("Chronological sequence values must be unique")
    history = RiskHistory()
    rows = []
    for position, record in enumerate(sorted(records, key=lambda row: row["sequence"])):
        tests, outcomes = record["tests"], record["outcomes"]
        if len(tests) != len(set(tests)) or any(
            outcomes.get(n) not in OBSERVED_OUTCOMES for n in tests
        ):
            raise ValueError(
                "Replay needs a unique selected set with complete pass/fail outcomes"
            )
        order, scores = history.order(tests, record["files"])
        if position >= warmup:
            for policy, names in (("original", tests), ("fastci", order)):
                rank = next(
                    (i + 1 for i, name in enumerate(names) if outcomes[name] == "fail"),
                    None,
                )
                prefix = names[:rank] if rank else []
                seconds = record.get("seconds", {})
                gpu_counts = record.get("gpus", {})
                costs_available = rank is not None and all(
                    n in seconds and n in gpu_counts for n in prefix
                )
                rows.append(
                    {
                        "run_id": record["run_id"],
                        "policy": policy,
                        "order": names,
                        "first_failure_rank": rank,
                        cost_key: sum(seconds[n] * gpu_counts[n] for n in prefix)
                        if costs_available
                        else None,
                        "scores": scores if policy == "fastci" else {},
                    }
                )
        # Every MR, including no-failure MRs and warmup, contributes history.
        history.observe(
            record["run_id"], record["files"], {n: outcomes[n] for n in tests}
        )
    return {
        "kind": "offline_ordering_replay",
        "resource_mode": resource_mode,
        "warmup": warmup,
        "mr_count": len(records),
        "rows": rows,
    }

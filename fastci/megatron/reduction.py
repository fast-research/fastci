"""Conservative, recipe-driven shortening of deterministic training tests."""

from __future__ import annotations

import json
import math

from ..model import write_json
from .common import case_directory, file_hash, golden_filename

RULE = "deterministic-pretraining-prefix-v1"
NUMERICAL_METRICS = {"lm loss", "num-zeros"}
EXIT_ASSIGNMENT = '.MODEL_ARGS."--exit-interval" = .MODEL_ARGS."--train-iters"'


def prefix_metrics(golden, steps):
    prefix = {}
    for name in sorted(NUMERICAL_METRICS):
        metric = golden[name]
        values = {
            key: value for key, value in metric["values"].items() if int(key) <= steps
        }
        if values:
            prefix[name] = {
                **metric,
                "end_step": max(map(int, values)),
                "values": values,
            }
    return prefix


def reduction_for(recipe, golden, repeats):
    if (
        recipe.get("TEST_TYPE") != "regular"
        or recipe.get("MODE", "pretraining") != "pretraining"
    ):
        return None
    if repeats != 1 or recipe.get("BEFORE_SCRIPT") or recipe.get("AFTER_SCRIPT"):
        return None
    if any(key.startswith("MODEL_ARGS_") for key in recipe):
        return None

    args = recipe.get("MODEL_ARGS", {})
    env = recipe.get("ENV_VARS", {})
    metrics = recipe.get("METRICS", sorted(NUMERICAL_METRICS))
    if not isinstance(metrics, list) or sorted(map(str, metrics)) != sorted(
        NUMERICAL_METRICS
    ):
        return None
    if args.get("--deterministic-mode") is not True or any(
        str(env.get(key, "0")).lower() not in {"false", "0"}
        for key in ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "NON_DETERMINISTIC_RESULTS")
    ):
        return None
    if any(
        key.startswith(("--cuda-graph", "--profile", "--rampup", "--batch-size-ramp"))
        for key in args
    ):
        return None
    if any(
        key in args
        for key in (
            "--enable-cuda-graph",
            "--train-samples",
            "--non-persistent-save-interval",
            "--exit-duration-in-mins",
        )
    ):
        return None

    full_steps = args.get("--train-iters")
    if type(full_steps) is not int or full_steps < 50:
        return None
    if args.get("--exit-interval", full_steps) != full_steps:
        return None
    for key in ("--save-interval", "--eval-interval"):
        interval = args.get(key)
        if type(interval) is not int or interval <= full_steps:
            return None

    steps = (full_steps + 1) // 2
    try:
        data = json.loads(golden.read_text())
        prefix = prefix_metrics(data, steps)
        for name in metrics:
            values = prefix[name]["values"]
            if max(map(int, data[name]["values"])) != full_steps:
                return None
            if set(map(int, values)) != set(range(1, steps + 1)):
                return None
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in values.values()
            ):
                return None
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return None

    return {"rule": RULE, "full_steps": full_steps, "steps": steps}


def prepare_reduction(source, row, cfg):
    reduction = row.get("reduction")
    if not reduction:
        return
    golden = case_directory(source, row["model"], row["case"]) / golden_filename(cfg)
    if file_hash(golden) != row["golden_hash"]:
        raise ValueError("Golden values changed after planning")
    shell = source / "tests/functional_tests/shell_test_utils/run_ci_test.sh"
    script = shell.read_text()
    if script.count(EXIT_ASSIGNMENT) != 1:
        raise ValueError(
            "Unrecognized upstream exit-interval assignment for workload reduction"
        )
    steps = reduction["steps"]
    prefix = prefix_metrics(json.loads(golden.read_text()), steps)
    write_json(golden, prefix)
    shell.write_text(
        script.replace(EXIT_ASSIGNMENT, f'.MODEL_ARGS."--exit-interval" = {steps}')
    )
    return {
        **reduction,
        "source_golden_hash": row["golden_hash"],
        "prefix_golden_hash": file_hash(golden),
    }

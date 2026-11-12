from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from ..repository import resolve_commit
from .common import (
    GPUS_PER_NODE,
    case_directory,
    digest,
    export,
    file_hash,
    golden_filename,
    seal,
)
from .reduction import reduction_for


def config(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    allowed = {
        "scope",
        "cadence",
        "platform",
        "environment",
        "hardware",
        "environment_id",
        "python",
        "yq",
        "data_dir",
        "hf_home",
        "golden_dir",
        "cases",
        "allowlist",
        "n_repeat",
        "timeout_seconds",
        "trace_rate_hz",
        "trace",
        "master_port",
        "workload_reduction",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    value = {
        "scope": "L1",
        "cadence": "pr",
        "environment": "dev",
        "hardware": "local",
        "n_repeat": 1,
        "timeout_seconds": 3600,
        "trace_rate_hz": 50,
        "trace": True,
        "python": sys.executable,
        "master_port": 29500,
        **value,
    }
    if value["environment"] != "dev":
        raise ValueError("The native adapter supports upstream dev recipes")
    if "platform" in value and not (
        isinstance(value["platform"], str)
        and re.fullmatch(r"[a-zA-Z0-9_-]+", value["platform"])
    ):
        raise ValueError("platform must be an upstream recipe label")
    if not all(
        isinstance(value.get(key), str) and value[key]
        for key in ("environment_id", "hardware")
    ):
        raise ValueError(
            "An explicit dependency environment_id and hardware label are required"
        )
    for key in ("n_repeat", "timeout_seconds", "trace_rate_hz", "master_port"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(value["trace"]) is not bool:
        raise ValueError("trace must be boolean")
    if type(value.get("workload_reduction", False)) is not bool:
        raise ValueError("workload_reduction must be boolean")
    for key in ("cases", "allowlist"):
        if key in value and (
            not isinstance(value[key], list)
            or not all(isinstance(x, str) and x for x in value[key])
        ):
            raise ValueError(f"{key} must be a list of nonempty test_case names")
    if not value.get("cases") or len(set(value["cases"])) != len(value["cases"]):
        raise ValueError(
            "Set cases to the complete, unique list of functional tests in scope"
        )
    if value["master_port"] > 65535:
        raise ValueError("master_port must be a valid TCP port")
    for key in ("data_dir", "hf_home", "golden_dir", "python", "yq"):
        if value.get(key):
            if not isinstance(value[key], str):
                raise ValueError(f"{key} must be a path")
            # Preserve the venv's Python symlink: resolving it selects the base interpreter.
            value[key] = os.path.abspath(path.parent / value[key])
    return value


def case_key(case):
    """Stable reference to an upstream case without embedding its recipe name."""
    return "case-" + digest(case)[:16]


def _recipe_specs(workloads, cfg):
    requested = set(cfg["cases"])
    by_platform = defaultdict(list)
    for workload in workloads:
        if workload["type"] == "build":
            continue
        item = workload["spec"]
        case = item["test_case"]
        if {case, case_key(case)} & requested:
            by_platform[item["platforms"]].append(item)
    candidates = []
    for platform, specs in by_platform.items():
        names = {item["test_case"] for item in specs}
        references = names | {case_key(case) for case in names}
        if requested <= references:
            candidates.append((platform, specs))
    if not candidates:
        raise ValueError(
            "No upstream recipe platform contains every requested case in the scope"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Cases match multiple recipe platforms; set platform from the upstream recipes"
        )
    platform, specs = candidates[0]
    cfg["platform"] = platform
    return specs


def _load_workloads(root, cfg, sha):
    parser_path = root / "tests/test_utils/python_scripts/recipe_parser.py"
    spec = importlib.util.spec_from_file_location(
        "_fastci_megatron_recipe_parser", parser_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_workloads(
        container_tag=sha,
        scope=cfg["scope"],
        cadence=cfg.get("cadence"),
        environment=cfg["environment"],
        platform=cfg.get("platform"),
        n_repeat=cfg["n_repeat"],
    )


def _catalog_row(root, cfg, spec):
    item = json.loads(json.dumps(spec))
    case, model = item["test_case"], item["model"]
    if not all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in (model, case)):
        raise ValueError("Invalid Megatron test case or model name")
    if model == "unit-tests":
        raise ValueError("Unit tests remain owned by the existing Testmon workflow")
    if item.get("nodes") != 1 or item.get("gpus") != GPUS_PER_NODE:
        raise ValueError(f"{case}: only single-node, eight-GPU workloads are supported")
    case_dir = case_directory(root, model, case)
    params = case_dir / "model_config.yaml"
    if not params.is_file():
        raise ValueError(
            f"{case}: no model_config.yaml; custom recipe scripts need an adapter"
        )
    if "shell_test_utils/run_ci_test.sh" not in item.get("script", ""):
        raise ValueError(f"{case}: not a standard functional-test script")
    params_data = yaml.safe_load(params.read_text())
    test_type = params_data.get("TEST_TYPE")
    if (
        test_type not in {"regular", "ckpt-resume"}
        or params_data.get("MODE", "pretraining") != "pretraining"
    ):
        raise ValueError(
            f"{case}: native runner supports regular and checkpoint-resume pretraining"
        )
    env = params_data.get("ENV_VARS", {})
    if env.get("ENABLE_LIGHTWEIGHT_MODE") in (True, "true"):
        raise ValueError(f"{case}: lightweight mode is not a correctness baseline")
    if env.get("SKIP_PYTEST") in (1, "1", True):
        raise ValueError(f"{case}: pytest validation is disabled")
    override = env.get("N_REPEAT", cfg["n_repeat"])
    if str(override) != str(cfg["n_repeat"]):
        raise ValueError(
            f"{case}: model_config overrides n_repeat; match the explicit experiment setting"
        )
    if any(key.startswith("MODEL_ARGS_") for key in params_data):
        raise ValueError(
            f"{case}: multi-phase model configurations need an explicit adapter"
        )
    golden_dir = Path(cfg["golden_dir"]) if cfg.get("golden_dir") else None
    golden_parent = golden_dir / model / case if golden_dir else case_dir
    golden = golden_parent / golden_filename(cfg)
    if not golden.is_file():
        raise ValueError(f"{case}: missing golden values: {golden}")
    golden_hash = file_hash(golden)
    identity = "/".join(
        [model, case, cfg["platform"], cfg["environment"], cfg["hardware"]]
    )
    reduction = (
        reduction_for(params_data, golden, cfg["n_repeat"])
        if cfg.get("workload_reduction")
        else None
    )
    # Container metadata describes upstream scheduling. The native runner
    # uses the expanded recipe script and the prepared local environment.
    native_spec = {
        key: value
        for key, value in item.items()
        if key
        not in {"container_tag", "container_image", "image", "build", "script_setup"}
    }
    validation_key = digest(
        {
            "model_config": file_hash(params),
            "golden": golden_hash,
            "script": item["script"],
            "n_repeat": cfg["n_repeat"],
        }
    )
    fingerprint = digest(
        {
            "spec": native_spec,
            "validation": validation_key,
            "environment_id": cfg["environment_id"],
            "hardware": cfg["hardware"],
            "n_repeat": cfg["n_repeat"],
            **({"reduction": reduction} if reduction else {}),
        }
    )
    return {
        "id": identity,
        "case": case,
        "model": model,
        "spec": native_spec,
        "validation_key": validation_key,
        **({"training_phases": 2} if test_type == "ckpt-resume" else {}),
        **({"reduction": reduction} if reduction else {}),
        "launcher": params_data.get("LAUNCHER", "torchrun"),
        "fingerprint": fingerprint,
        "golden_hash": golden_hash,
        "golden_relative": str(golden.relative_to(golden_dir)) if golden_dir else None,
    }


def _load(root, cfg, sha):
    specs = _recipe_specs(_load_workloads(root, cfg, sha), cfg)
    rows = [_catalog_row(root, cfg, spec) for spec in specs]
    names = [row["case"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Recipe expansion produced ambiguous duplicate test cases")
    if len(names) != len(cfg["cases"]):
        raise ValueError("A case was requested by both name and reference")
    if cfg.get("allowlist") is not None:
        references = {
            reference: case for case in names for reference in (case, case_key(case))
        }
        if not set(cfg["allowlist"]) <= references.keys():
            raise ValueError("allowlist must be a subset of the baseline cases")
        cfg["allowlist"] = [references[reference] for reference in cfg["allowlist"]]
    cfg["cases"] = names
    return rows


def catalog(repo, cfg, source="HEAD"):
    sha = resolve_commit(repo, source)
    cfg = dict(cfg)
    with tempfile.TemporaryDirectory(prefix="fastci-megatron-catalog-") as directory:
        root = Path(directory) / "source"
        export(repo, sha, root)
        rows = _load(root, cfg, sha)
    return seal({"schema": 1, "revision": sha, "config": cfg, "tests": rows})

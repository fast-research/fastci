from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from ..model import write_json
from .common import (
    GPUS_PER_NODE,
    MODES,
    RANKS,
    case_directory,
    digest,
    export,
    file_hash,
    seal,
    trace_phases,
    verify,
)
from .reduction import prepare_reduction

ENVIRONMENT_PROBE = """
import importlib.metadata as m, json, platform, sys, torch
print(json.dumps({"python": sys.version, "machine": platform.machine(),
  "packages": sorted((d.metadata['Name'], d.version) for d in m.distributions() if d.metadata['Name']),
  "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
  "gpu_count": torch.cuda.device_count()}))
"""


def environment(cfg):
    env = dict(os.environ)
    for key in list(env):
        if key in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
        } or key.startswith("FASTCI_MEGATRON_"):
            env.pop(key)
    env.update(
        PYTHONUNBUFFERED="1",
        ENABLE_LIGHTWEIGHT_MODE="false",
        RECORD_CHECKPOINTS="false",
        N_REPEAT=str(cfg["n_repeat"]),
        NUM_NODES="1",
        NODE_RANK="0",
        GPUS_PER_NODE=str(GPUS_PER_NODE),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(cfg["master_port"]),
        NVTE_FLASH_ATTN_V4="0",
        NCCL_DEBUG="INFO",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES=",".join(map(str, RANKS)),
    )
    env["PATH"] = str(Path(cfg["python"]).parent) + os.pathsep + env.get("PATH", "")
    if cfg.get("hf_home"):
        env["HF_HOME"] = cfg["hf_home"]
    return env


def probe(cfg):
    if "REPLACE_WITH" in cfg["environment_id"] or not Path(cfg["python"]).is_file():
        raise ValueError(
            "Configure the prepared Python environment and environment_id first"
        )
    if not cfg.get("data_dir") or not Path(cfg["data_dir"]).is_dir():
        raise ValueError("An existing data_dir is required")
    if cfg.get("hf_home") and not Path(cfg["hf_home"]).is_dir():
        raise ValueError("hf_home does not exist")
    env = environment(cfg)
    tools = ["bash", "git", "tar", "envsubst", "realpath", "nvidia-smi"]
    if cfg["trace"]:
        tools.append("py-spy")
    for tool in tools:
        if not shutil.which(tool, path=env["PATH"]):
            raise ValueError(f"Missing native runtime tool: {tool}")
    yq = cfg.get("yq") or shutil.which("yq", path=env["PATH"])
    if not yq:
        raise ValueError("Install Mike Farah yq v4 and set yq in the configuration")
    version = subprocess.check_output([yq, "--version"], text=True)
    if not re.search(r"version v?4\.", version):
        raise ValueError("Megatron's native scripts require Mike Farah yq v4")
    info = json.loads(
        subprocess.check_output(
            [cfg["python"], "-c", ENVIRONMENT_PROBE], env=env, text=True
        )
    )
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    )
    devices = list(csv.reader(io.StringIO(hardware)))
    names = {d[0].strip() for d in devices}
    if len(devices) != GPUS_PER_NODE or len(names) != 1:
        raise ValueError("Runner requires a dedicated host with eight homogeneous GPUs")
    if (
        cfg["hardware"] != "local"
        and cfg["hardware"].lower() not in next(iter(names)).lower()
    ):
        raise ValueError("The host GPUs do not match the configured hardware label")
    if not info["cuda_available"] or info["gpu_count"] != GPUS_PER_NODE:
        raise ValueError(
            "The configured Python environment cannot access all eight CUDA GPUs"
        )
    package = Path(__file__).resolve().parents[1]
    adapter_files = [
        *(package / "megatron").rglob("*.py"),
        package / "repository.py",
        package / "source.py",
        package / "traces.py",
    ]
    compatibility = {
        **info,
        "gpu_models": [d[0].strip() for d in devices],
        "drivers": [d[2].strip() for d in devices],
        "yq": version.strip(),
        "runtime_settings": {
            k: v
            for k, v in env.items()
            if k.startswith(
                ("NCCL_", "TORCH_", "NVTE_", "CUDA_", "CUBLAS_", "OMP_", "MKL_")
            )
        },
        "adapter": digest(
            {str(p.relative_to(package)): file_hash(p) for p in adapter_files}
        ),
    }
    return {
        "details": compatibility,
        "fingerprint": digest(compatibility),
        "gpu_uuids": [d[1].strip() for d in devices],
        "yq": str(Path(yq).resolve()),
    }


def script(row, cfg, source):
    values = dict(row["spec"])
    values.update(
        assets_dir=str(source / "assets_dir"),
        artifacts_dir=str(source / "artifacts_dir"),
        n_repeat=cfg["n_repeat"],
    )
    values["name"] = values["name"].format(**values)
    rendered = values["script"].format(**values)
    # Recipes clean up container credentials. Keep that cleanup inside the
    # disposable export when running on a user's native host.
    rendered = rendered.replace(
        '"${HOME}/.netrc"', '"' + str(source / "assets_dir/.netrc") + '"'
    )
    rendered = rendered.replace("/opt/megatron-lm", str(source))
    rendered = rendered.replace(
        "/mnt/artifacts", cfg.get("data_dir") or "/DATA_PATH_NOT_CONFIGURED"
    )
    rendered = rendered.replace(
        "/workspace/data/cache", str(source / "assets_dir/data-cache")
    )
    return "set -euo pipefail\ncd " + str(source) + "\n" + rendered + "\n"


def prepare_native_scripts(source, cfg, yq):
    """Rebind paths in the disposable export; retain training args and checks."""
    for name in ("run_ci_test.sh", "_run_training.sh"):
        path = source / "tests/functional_tests/shell_test_utils" / name
        text = path.read_text()
        text = text.replace("/usr/local/bin/yq", yq)
        text = text.replace("/opt/venv/bin", str(Path(cfg["python"]).parent))
        text = text.replace(
            "/tmp/checkpoints", str(source / "artifacts_dir/empty-load")
        )
        text = text.replace(
            "export PYTHONPATH=$ROOT_DIR",
            "export PYTHONPATH=$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}",
        )
        text = text.replace("uv run --no-sync python", cfg["python"])
        text = text.replace("uv run --no-sync pytest", cfg["python"] + " -m pytest")
        if "uv run" in text:
            raise ValueError(
                f"{name}: unrecognized uv invocation; update native adaptation explicitly"
            )
        path.write_text(text)
    makefile = source / "megatron/core/datasets/Makefile"
    if makefile.is_file():
        text = makefile.read_text()
        text = text.replace(
            "$(shell python3 -m pybind11 --includes)",
            "$(shell " + cfg["python"] + " -m pybind11 --includes)",
        )
        text = text.replace(
            "$(shell python3-config --extension-suffix)",
            "$(shell "
            + cfg["python"]
            + " -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))')",
        )
        makefile.write_text(text)


def execute(argv, log, timeout, env, cwd):
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            argv,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            cwd=cwd,
        )
        try:
            return process.wait(timeout=timeout), False
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if isinstance(error, KeyboardInterrupt):
                raise
            return 124, True


def trace_manifest(source, n_repeat=1, training_phases=1):
    directory = source / "assets_dir/fastci-traces"
    paths = sorted(directory.glob("*.json"))
    entries, ranks, errors = [], set(), []
    coverage = set()
    expected_phases = set(trace_phases(n_repeat, training_phases))
    for path in list(directory.glob("*.speedscope")) + list(directory.glob("*.log")):
        if not path.with_suffix(".json").is_file():
            errors.append(f"{path.name}: missing sampler completion report")
    for path in paths:
        try:
            report = json.loads(path.read_text())
            rank = report["rank"]
            if (
                type(rank) is not int
                or rank not in RANKS
                or report["world_size"] != GPUS_PER_NODE
            ):
                raise ValueError("unexpected rank topology")
            if not report["complete"]:
                raise ValueError("sampler did not finish successfully")
            phase = (report.get("repeat", 1), report.get("phase", 1))
            if phase not in expected_phases:
                raise ValueError(f"unexpected repeat and phase: {phase}")
            trace = path.with_suffix(".speedscope")
            data = json.loads(trace.read_text())
            if not any(
                p.get("type") == "sampled" and p.get("samples")
                for p in data.get("profiles", [])
            ):
                raise ValueError("trace has no sampled stacks")
            entries.append(
                {
                    "rank": rank,
                    "repeat": phase[0],
                    "phase": phase[1],
                    "path": str(trace.relative_to(source.parent)),
                    "sha256": file_hash(trace),
                }
            )
            ranks.add(rank)
            if (*phase, rank) in coverage:
                errors.append(
                    f"{path.name}: duplicate sampler for repeat, phase and rank"
                )
            coverage.add((*phase, rank))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.name}: {error}")
    if ranks != set(RANKS):
        errors.append(f"incomplete rank coverage: {sorted(ranks)}")
    for repeat, phase in sorted(expected_phases):
        missing = [rank for rank in RANKS if (repeat, phase, rank) not in coverage]
        if missing:
            errors.append(f"repeat {repeat}, phase {phase}: missing ranks {missing}")
    return entries, errors


def run_case(
    repo,
    revision,
    row,
    cfg,
    runtime,
    output,
    directory,
    dry_run,
    executor,
    prefix=False,
):
    source = directory / "source"
    export(repo, revision, source)
    (source / "artifacts_dir").mkdir(exist_ok=True)
    (source / "artifacts_dir/empty-load").mkdir()
    (source / "assets_dir/data-cache").mkdir(parents=True, exist_ok=True)
    if row.get("golden_relative"):
        golden = Path(cfg["golden_dir"]) / row["golden_relative"]
        if file_hash(golden) != row["golden_hash"]:
            raise ValueError("Golden values changed after planning")
        destination = case_directory(source, row["model"], row["case"]) / golden.name
        shutil.copyfile(golden, destination)
    yq = runtime["yq"] if runtime else cfg.get("yq") or "/usr/local/bin/yq"
    prepare_native_scripts(source, cfg, yq)
    reduction = prepare_reduction(source, row, cfg) if prefix else None
    (source / "fastci-job.sh").write_text(script(row, cfg, source))
    env = environment(cfg)
    env["PYTHONPATH"] = str(source)
    if cfg["trace"] and not prefix:
        env.update(
            PYTHONPATH=str(Path(__file__).parent / "trace_hook")
            + os.pathsep
            + str(source),
            FASTCI_MEGATRON_TRACE="1",
            FASTCI_MEGATRON_TRACE_DIR=str(source / "assets_dir/fastci-traces"),
            FASTCI_MEGATRON_TRACE_RATE=str(cfg["trace_rate_hz"]),
            FASTCI_MEGATRON_SOURCE_ROOT=str(source),
        )
    argv = ["bash", str(source / "fastci-job.sh")]
    log = directory / "stdout.log"
    started = time.monotonic()
    code, timed_out = (
        (None, False)
        if dry_run
        else executor(argv, log, cfg["timeout_seconds"], env, source)
    )
    seconds = 0.0 if dry_run else time.monotonic() - started
    traces, errors = (
        trace_manifest(source, cfg["n_repeat"], row.get("training_phases", 1))
        if cfg["trace"] and not dry_run and not prefix
        else ([], ["dry_run" if dry_run else "sampling_disabled"])
    )
    return {
        "test": row["id"],
        **({"reduction": reduction} if reduction else {}),
        "exit_code": code,
        "status": "not_run"
        if dry_run
        else ("timeout" if timed_out else "passed" if code == 0 else "failed"),
        "seconds": seconds,
        "gpu_seconds": None if dry_run else GPUS_PER_NODE * seconds,
        "source_root": str(source),
        "traces": [
            {**trace, "path": str(directory.relative_to(output) / trace["path"])}
            for trace in traces
        ],
        "trace_errors": errors,
        "command": argv,
        "log": str(log.relative_to(output)),
    }


def _validate_execution(plan, cfg, rows, output):
    # Upstream scripts contain unquoted expansions; reject unsupported shell paths.
    for path in [
        str(output),
        cfg["python"],
        cfg.get("data_dir", ""),
        cfg.get("hf_home", ""),
        cfg.get("yq", ""),
    ]:
        if re.search(r"[^a-zA-Z0-9_./:+-]", path):
            raise ValueError(
                "Native Megatron paths must omit whitespace and shell metacharacters"
            )
    if plan["mode"] not in MODES or plan["baseline"] != list(rows):
        raise ValueError("Invalid execution mode or baseline")
    proposed = plan["proposed"]
    if (
        len(set(proposed)) != len(proposed)
        or not proposed
        or set(proposed) - rows.keys()
    ):
        raise ValueError("Invalid proposed tests")
    expected = proposed if plan["mode"] == "enforce" else plan["baseline"]
    if plan["execute"] != expected:
        raise ValueError("Execution order is inconsistent with the mode")


def run(repo, plan, output, dry_run=False, executor=execute, environment_probe=probe):
    verify(plan)
    catalog = plan["catalog"]
    verify(catalog)
    cfg = catalog["config"]
    rows = {row["id"]: row for row in catalog["tests"]}
    output = Path(output).resolve()
    _validate_execution(plan, cfg, rows, output)
    runtime = None if dry_run else environment_probe(cfg)
    base = plan.get("base_evidence")
    if runtime and base and base.get("runtime_fingerprint") != runtime["fingerprint"]:
        plan = seal(
            dict(
                plan,
                proposed=plan["baseline"],
                execute=plan["baseline"],
                workload_reduction=False,
                base_evidence=None,
                fallback_reasons=plan["fallback_reasons"]
                + ["runtime_environment_changed"],
            )
        )
    if not dry_run and any(
        rows[name]["launcher"] == "ft_launcher" for name in plan["execute"]
    ):
        if not shutil.which("ft_launcher", path=environment(cfg)["PATH"]):
            raise ValueError(
                "Selected tests require ft_launcher in the prepared environment"
            )
    if runtime and re.search(r"[^a-zA-Z0-9_./:+-]", runtime["yq"]):
        raise ValueError(
            "Resolved yq path is unsupported by the upstream shell scripts"
        )
    output.mkdir(parents=True, exist_ok=False)
    import fcntl

    lock = None if dry_run else open("/tmp/fastci-megatron-gpu.lock", "a")
    try:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcomes = []
        for name in plan["execute"]:
            row = rows[name]
            directory = output / digest(name)[:16]
            outcome = run_case(
                repo,
                catalog["revision"],
                row,
                cfg,
                runtime,
                output,
                directory,
                dry_run,
                executor,
            )
            if plan["workload_reduction"] and row.get("reduction"):
                # Keep the full observation authoritative during prefix validation.
                try:
                    prefix = run_case(
                        repo,
                        catalog["revision"],
                        row,
                        cfg,
                        runtime,
                        output,
                        directory / "prefix",
                        dry_run,
                        executor,
                        prefix=True,
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    prefix = {
                        "status": "unavailable",
                        "error": str(error),
                        "gpu_seconds": None,
                    }
                outcome["prefix"] = prefix
            outcomes.append(outcome)
            write_json(output / "progress.json", outcomes)
        gpu_seconds = (
            None
            if dry_run
            else sum(
                row["gpu_seconds"] + (row.get("prefix", {}).get("gpu_seconds") or 0)
                for row in outcomes
            )
        )
        result = seal(
            {
                "schema": 1,
                "kind": "dry_run" if dry_run else "native_gpu_run",
                "revision": catalog["revision"],
                "plan": plan,
                "source_repo": str(Path(repo).resolve()),
                "runtime": runtime,
                "outcomes": outcomes,
                "gpu_seconds": gpu_seconds,
                "all_passed": not dry_run
                and all(row["status"] == "passed" for row in outcomes),
            }
        )
        write_json(output / "run.json", result)
        return result
    finally:
        if lock is not None:
            lock.close()

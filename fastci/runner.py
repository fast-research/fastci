from __future__ import annotations

import concurrent.futures as futures
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .model import TestSpec, digest, file_hash
from .workload import Proposal, WorkloadGuard, invocation


@dataclass
class Outcome:
    test: str
    status: str
    attempts: list[dict] = field(default_factory=list)
    queued_seconds: float = 0
    devices: list[str] = field(default_factory=list)
    workload_action: str = "original"
    comparison_agreed: bool | None = None
    started_seconds: float | None = None
    finished_seconds: float | None = None


def _configure_trace(env: dict, trace: dict) -> None:
    trace["directory"].mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parent
    env.update(
        FASTCI_TRACE="1" if trace["enabled"] else "0",
        FASTCI_TRACE_DIR=str(trace["directory"]),
        FASTCI_TRACE_RATE=str(trace["rate"]),
        FASTCI_TRACE_SUBPROCESSES="1" if trace["subprocesses"] else "0",
    )
    if trace.get("config"):
        env["FASTCI_CONFIG"] = str(Path(trace["config"]).resolve())
    if trace["enabled"]:
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(package / "_trace_hook"),
                str(package.parent),
                env.get("PYTHONPATH", ""),
            ]
        )


def _capture_trace_files(trace: dict) -> tuple[list[dict], list[str]]:
    captures = [
        {"path": str(path.resolve()), "sha256": file_hash(path)}
        for path in sorted(trace["directory"].glob("*.speedscope"))
    ]
    if not captures and trace["enabled"]:
        return captures, [
            "No trace produced; inspect sampler logs or runner tracing support"
        ]
    return captures, []


def execute(
    test: TestSpec,
    overrides: dict,
    devices: list[str],
    cwd: Path,
    log: Path,
    label: str,
    trace: dict | None = None,
) -> dict:
    command, extra = invocation(test, overrides)
    env = {**os.environ, **extra, "CUDA_VISIBLE_DEVICES": ",".join(devices)}
    if trace is not None:
        _configure_trace(env, trace)
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    code, status, error = None, "error", None
    with log.open("wb") as output:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=test.timeout)
                status = "pass" if code == 0 else "fail"
            except subprocess.TimeoutExpired:
                status = "timeout"
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        except OSError as exc:
            error = str(exc)
    trace_files, trace_errors = (
        _capture_trace_files(trace) if trace is not None else ([], [])
    )
    return {
        "mode": label,
        "status": status,
        "exit_code": code,
        "seconds": time.monotonic() - start,
        "gpus": len(devices),
        "log": str(log.resolve()),
        "overrides": overrides,
        "error": error,
        "trace_files": trace_files,
        "trace_errors": trace_errors,
    }


def _validate_run(order, suite, proposals, pools, workers):
    if workers < 1 or len(order) != len(set(order)):
        raise ValueError("Positive workers and unique test names required")
    if set(order) - suite.keys() or set(order) - proposals.keys():
        raise ValueError("Plan contains unknown tests or missing proposals")
    all_devices = [device for devices in pools.values() for device in devices]
    if len(set(all_devices)) != len(all_devices):
        raise ValueError("Device IDs must be unique across pools")
    for name in order:
        test = suite[name]
        if test.metadata.get("resources_unresolved"):
            raise ValueError(f"Annotate hardware requirements before running {name}")
        if not test.command:
            raise ValueError(
                f"No runnable argv for {name}; configure its existing CI runner"
            )
        if test.gpus > len(pools.get(test.hardware, [])):
            raise ValueError(f"Insufficient {test.hardware} devices for {name}")
        invocation(test, proposals[name].overrides)


def _authoritative_status(
    name, attempts, action, guard, proposal, environment, revision
):
    if action == "paired":
        original, reduced = (attempt["status"] for attempt in attempts)
        agreed = guard.compare(name, proposal, environment, original, reduced, revision)
        # On disagreement, retain the original outcome and account for both attempts.
        return (reduced if agreed else original), agreed
    if action == "reduced":
        guard.used(name, proposal, environment)
    return attempts[0]["status"], None


def run_tests(
    order: list[str],
    suite: dict[str, TestSpec],
    proposals: dict[str, Proposal],
    guard: WorkloadGuard,
    environment: str,
    revision: str,
    cwd: str | Path,
    pools: dict[str, list[str]],
    log_dir: str | Path,
    workers: int = 4,
    stop_on_failure: bool = False,
    trace_dir: str | Path | None = None,
    trace_enabled: bool = True,
    trace_rate: int = 50,
    trace_subprocesses: bool = False,
    verify_source: Callable[[], None] | None = None,
    trace_config: str | None = None,
) -> list[Outcome]:
    """Priority-ordered, first-fitting dispatch to homogeneous local device pools.

    Resource packing is infrastructure, not a novel algorithm from the paper.
    Failure stops pending work only; already running jobs complete and are counted.
    """
    available = {label: list(devices) for label, devices in pools.items()}
    _validate_run(order, suite, proposals, available, workers)
    if trace_rate < 1:
        raise ValueError("Trace rate must be positive")
    if trace_dir is not None:
        trace_dir = Path(trace_dir).resolve()
        trace_dir.mkdir(parents=True, exist_ok=False)
    root, cwd = Path(log_dir), Path(cwd)
    if not cwd.is_dir():
        raise ValueError("Runner cwd does not exist")
    root.mkdir(parents=True, exist_ok=True)
    # Prevent overwriting evidence/logs of a prior run.
    lock = root / "run.started"
    with lock.open("x") as handle:
        handle.write(revision + "\n")
    start = time.monotonic()
    pending = list(order)
    running: dict[futures.Future, tuple[str, list[str], str, float]] = {}
    results: dict[str, Outcome] = {}
    stopped = False

    def work(name: str, devices: list[str], action: str) -> list[dict]:
        test, proposal = suite[name], proposals[name]
        test_id = digest(name)[:16]
        folder = root / test_id

        def attempt(overrides, label):
            if verify_source is not None:
                verify_source()
            settings = (
                {
                    "directory": trace_dir / test_id / label,
                    "enabled": trace_enabled,
                    "rate": trace_rate,
                    "subprocesses": trace_subprocesses,
                }
                if trace_dir
                else None
            )
            if settings is not None:
                settings["config"] = trace_config
            result = execute(
                test,
                overrides,
                devices,
                cwd,
                folder / (label + ".log"),
                label,
                settings,
            )
            if verify_source is not None:
                verify_source()
            return result

        if action == "paired":
            # Both use the same checkout, command, allocation and environment.
            return [attempt({}, "original"), attempt(proposal.overrides, "reduced")]
        overrides = proposal.overrides if action == "reduced" else {}
        return [attempt(overrides, action)]

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while pending or running:
            if not stopped:
                for name in list(pending):
                    if len(running) >= workers:
                        break
                    test = suite[name]
                    devices = available.get(test.hardware, [])
                    if len(devices) < test.gpus:
                        continue
                    allocated = devices[: test.gpus]
                    del devices[: test.gpus]
                    action = guard.action(name, proposals[name], environment)
                    waited = time.monotonic() - start
                    running[executor.submit(work, name, allocated, action)] = (
                        name,
                        allocated,
                        action,
                        waited,
                    )
                    pending.remove(name)
            if not running:
                break
            done, _ = futures.wait(running, return_when=futures.FIRST_COMPLETED)
            for future in done:
                name, allocated, action, waited = running.pop(future)
                available.setdefault(suite[name].hardware, []).extend(allocated)
                attempts = future.result()
                status, agree = _authoritative_status(
                    name,
                    attempts,
                    action,
                    guard,
                    proposals[name],
                    environment,
                    revision,
                )
                results[name] = Outcome(
                    name,
                    status,
                    attempts,
                    waited,
                    allocated,
                    action,
                    agree,
                    waited,
                    time.monotonic() - start,
                )
                if status != "pass" and stop_on_failure:
                    stopped = True
        for name in pending:
            results[name] = Outcome(name, "skipped")
    return [results[name] for name in order]


def report(outcomes: list[Outcome], resource_mode: str = "local") -> dict:
    if resource_mode not in ("local", "mock"):
        raise ValueError("Resource mode must be local or mock")
    device_seconds = sum(a["seconds"] * a["gpus"] for o in outcomes for a in o.attempts)
    reserved_seconds = sum(
        (o.finished_seconds - o.started_seconds) * len(o.devices)
        for o in outcomes
        if o.started_seconds is not None and o.finished_seconds is not None
    )
    return {
        "outcomes": [asdict(o) for o in outcomes],
        "resource_mode": resource_mode,
        "measured_gpu_seconds": device_seconds if resource_mode == "local" else None,
        "mock_device_seconds": device_seconds if resource_mode == "mock" else None,
        "reserved_gpu_seconds": reserved_seconds if resource_mode == "local" else None,
        "mock_reserved_device_seconds": reserved_seconds
        if resource_mode == "mock"
        else None,
        "wall_seconds": max((o.finished_seconds or 0 for o in outcomes), default=0),
        "attempt_seconds": sum(a["seconds"] for o in outcomes for a in o.attempts),
        "paired_tests": sum(o.workload_action == "paired" for o in outcomes),
        "all_passed": all(o.status == "pass" for o in outcomes),
    }

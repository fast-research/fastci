from __future__ import annotations

import atexit
import importlib.machinery
import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

from .policy import configured_mode


def install_hook(hook_file: str) -> None:
    """Preserve an existing application sitecustomize before adding sampling."""
    own = {
        Path(hook_file).resolve().parent,
        Path(__file__).resolve().parent / "_trace_hook",
        Path(__file__).resolve().parent.parent / "trace_hook",
    }
    search = [p for p in sys.path if Path(p or os.getcwd()).resolve() not in own]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", search)
    if spec is not None and spec.loader is not None:
        application = importlib.util.module_from_spec(spec)
        sys.modules["sitecustomize"] = application
        spec.loader.exec_module(application)
    start_rank0_sampler()


def sampler_command(
    pid: int, output: str | Path, rate: int = 50, subprocesses: bool = False
) -> list[str]:
    if pid < 1 or rate < 1:
        raise ValueError("Positive PID and sample rate required")
    command = [
        os.environ.get("FASTCI_PYSPY", "py-spy"),
        "record",
        "--pid",
        str(pid),
        "--rate",
        str(rate),
        "--format",
        "speedscope",
        "--output",
        str(output),
    ]
    if subprocesses:
        command.append("--subprocesses")
    return command


def start_rank0_sampler() -> None:
    """Opt-in launch hook for rank-0 Python processes.

    The existing runner sets PYTHONPATH to trace_hook + package root and
    FASTCI_TRACE_DIR to a per-test directory. Failure to trace is observable;
    it never changes test exit status or replaces old graph evidence.
    """
    if (
        os.environ.get("FASTCI_TRACE") != "1"
        or os.environ.get("LOCAL_RANK", "0") != "0"
    ):
        return
    if configured_mode(os.environ.get("FASTCI_CONFIG")) == "disabled":
        return
    root = Path(os.environ["FASTCI_TRACE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    path = root / f"rank0-{pid}.speedscope"
    command = sampler_command(
        pid,
        path,
        int(os.environ.get("FASTCI_TRACE_RATE", "50")),
        os.environ.get("FASTCI_TRACE_SUBPROCESSES") == "1",
    )
    error = (root / f"rank0-{pid}.sampler.log").open("wb")
    try:
        proc = subprocess.Popen(
            command, stdout=error, stderr=error, env={**os.environ, "FASTCI_TRACE": "0"}
        )
    except OSError as exc:
        error.close()
        print(f"[FastCI] sampling unavailable: {exc}", file=sys.stderr)
        return

    def finish() -> None:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        error.close()
        if not path.exists():
            print(f"[FastCI] missing trace: {path}", file=sys.stderr)

    atexit.register(finish)

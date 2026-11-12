"""Opt-in, per-rank py-spy hook; enabled for native Megatron training."""

import atexit
import json
import os
import signal
import subprocess
import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path


def start():
    if os.environ.get("FASTCI_MEGATRON_TRACE") != "1" or "RANK" not in os.environ:
        return
    # Multiprocessing's resource tracker and forkserver inherit RANK, but run
    # Python with -c. Their children may exit through os._exit without atexit.
    # Sample only directly launched repository scripts, not these helpers.
    root = os.environ.get("FASTCI_MEGATRON_SOURCE_ROOT")
    if not root or not sys.argv or not sys.argv[0].endswith(".py"):
        return
    try:
        Path(sys.argv[0]).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return
    rank = int(os.environ["RANK"])
    directory = Path(os.environ["FASTCI_MEGATRON_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"rank-{rank}-pid-{os.getpid()}"
    trace = str(stem) + ".speedscope"
    report_path = Path(str(stem) + ".json")
    report = {
        "rank": rank,
        "pid": os.getpid(),
        "repeat": int(os.environ.get("REPEAT", "1")),
        "phase": int(os.environ.get("RUN_NUMBER", "1")),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "trace": trace,
        "complete": False,
    }
    log = open(str(stem) + ".log", "w")
    try:
        if sys.platform.startswith("linux"):
            # Yama commonly forbids a child sampler attaching to its parent. Permit
            # tracing of this opt-in training process without requiring root/Docker.
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl.argtypes = [
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            if libc.prctl(0x59616D61, ctypes.c_ulong(-1).value, 0, 0, 0) != 0:
                report["ptrace_warning"] = os.strerror(ctypes.get_errno())
        process = subprocess.Popen(
            [
                "py-spy",
                "record",
                "--pid",
                str(os.getpid()),
                "--rate",
                os.environ["FASTCI_MEGATRON_TRACE_RATE"],
                "--format",
                "speedscope",
                "-o",
                trace,
            ],
            stdout=log,
            stderr=log,
        )
    except OSError as error:
        report["error"] = str(error)
        report_path.write_text(json.dumps(report))
        log.close()
        return

    def finish():
        try:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            code = process.wait(timeout=15)
            report.update(
                returncode=code, complete=code in (0, 130) and Path(trace).is_file()
            )
        except Exception as error:
            process.kill()
            process.wait()
            report["error"] = str(error)
        finally:
            log.close()
            report_path.write_text(json.dumps(report))

    atexit.register(finish)


try:
    # Preserve initialization supplied by the prepared environment when it has its own sitecustomize.
    own_directory = Path(__file__).parent.resolve()
    search = [p for p in sys.path if Path(p or ".").resolve() != own_directory]
    original = PathFinder.find_spec("sitecustomize", search)
    if original and original.loader:
        original.loader.exec_module(module_from_spec(original))
    start()
except Exception as error:
    # Instrumentation failure must not turn a failing/passing training test into another outcome.
    os.write(2, f"[FastCI] sampling unavailable: {error}\n".encode())

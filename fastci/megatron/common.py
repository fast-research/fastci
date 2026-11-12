from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..model import digest as digest
from ..model import file_hash as file_hash
from ..repository import git as run_git

MODES = ("baseline", "shadow", "enforce")
GPUS_PER_NODE = 8
RANKS = range(GPUS_PER_NODE)


def trace_phases(repeats, phases):
    return (
        (repeat, phase)
        for repeat in range(1, repeats + 1)
        for phase in range(1, phases + 1)
    )


def git(repo, *args):
    return run_git(repo, *args).strip()


def export(repo, sha, destination):
    """Export tracked files without checking out or modifying the user's tree."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # Extract a locally produced git archive, never an untrusted downloaded tar.
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "-C", str(repo), "archive", sha], stdout=archive, check=True
        )
        archive.seek(0)
        subprocess.run(
            ["tar", "-x", "-f", "-", "-C", str(destination)], stdin=archive, check=True
        )


def case_directory(source, model, case):
    return source / "tests/functional_tests/test_cases" / model / case


def golden_filename(cfg):
    return f"golden_values_{cfg['environment']}_{cfg['platform']}.json"


def seal(value):
    value = dict(value)
    value.pop("digest", None)
    return dict(value, digest=digest(value))


def verify(value):
    if not isinstance(value, dict) or value.get("digest") != seal(value)["digest"]:
        raise ValueError("Artifact checksum mismatch")


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("Artifact path escapes its run directory")
    return path

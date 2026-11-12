"""Read immutable Git revisions without changing the user's checkout."""

from __future__ import annotations

import io
import subprocess
import tokenize
from pathlib import Path


def git(repo: str | Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    ).stdout


def resolve_commit(repo: str | Path, ref: str) -> str:
    return git(
        repo, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"
    ).strip()


def merge_base(repo: str | Path, revision: str, target: str) -> str:
    return git(repo, "merge-base", revision, target).strip()


def tree_id(repo: str | Path, revision: str) -> str:
    return git(repo, "rev-parse", revision + "^{tree}").strip()


def tracked_paths(repo: str | Path, ref: str) -> list[str]:
    return git(repo, "ls-tree", "-rz", "--name-only", ref).split("\0")[:-1]


def changed_paths(
    repo: str | Path, base: str, revision: str, *, detect_renames: bool = False
) -> list[str]:
    options = () if detect_renames else ("--no-renames",)
    return git(repo, "diff", *options, "--name-only", "-z", base, revision).split("\0")[
        :-1
    ]


def sources_at(repo: str | Path, ref: str) -> dict[str, str]:
    """Read a version without switching branches or importing project code."""
    paths = [path for path in tracked_paths(repo, ref) if path.endswith(".py")]
    # One cat-file process for a whole revision, instead of one subprocess/file.
    payload = "".join(f"{ref}:{p}\n" for p in paths).encode()
    output = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input=payload,
        capture_output=True,
        check=True,
    ).stdout
    result, offset = {}, 0
    for path in paths:
        end = output.index(b"\n", offset)
        header = output[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"Cannot read source blob: {path}")
        size = int(header[2])
        offset = end + 1
        raw = output[offset : offset + size]
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            result[path] = raw.decode(encoding)
        except (SyntaxError, UnicodeError):
            # Invalid source must trigger fallback, not silent source omission.
            result[path] = "\0"
        offset += size + 1
    return result

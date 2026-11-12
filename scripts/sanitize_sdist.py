"""Remove local ownership and timestamps from source distribution archives."""

from __future__ import annotations

import argparse
import copy
import gzip
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def validate_member(member: tarfile.TarInfo) -> None:
    name = PurePosixPath(member.name)
    if (
        name.is_absolute()
        or ".." in name.parts
        or not (member.isfile() or member.isdir())
    ):
        raise ValueError(f"Unsupported archive entry: {member.name}")


def check_archive(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(10)
    if (
        len(header) != 10
        or header[:2] != b"\x1f\x8b"
        or header[3] != 0
        or any(header[4:8])
    ):
        raise ValueError(
            "Expected a gzip header without filename, comments, or timestamp"
        )
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            validate_member(member)
            if (member.uid, member.gid, member.uname, member.gname, member.mtime) != (
                0,
                0,
                "",
                "",
                0,
            ):
                raise ValueError(
                    f"Local ownership or timestamp in archive entry: {member.name}"
                )
            if member.pax_headers.keys() - {"path", "size"}:
                raise ValueError(
                    f"Unexpected extended metadata in archive entry: {member.name}"
                )


def sanitize_archive(path: Path) -> None:
    """Atomically rewrite a .tar.gz without changing its file contents."""
    if not path.name.endswith(".tar.gz"):
        raise ValueError("Expected a .tar.gz source distribution")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as raw, tarfile.open(path, "r:gz") as source:
            # An empty filename prevents embedding the temporary file's name.
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as target:
                    for member in sorted(
                        source.getmembers(), key=lambda entry: entry.name
                    ):
                        validate_member(member)
                        clean = copy.copy(member)
                        clean.uid = clean.gid = clean.mtime = 0
                        clean.uname = clean.gname = ""
                        clean.pax_headers = {}
                        clean.mode = (
                            0o755 if member.isdir() or member.mode & 0o111 else 0o644
                        )
                        content = (
                            source.extractfile(member) if member.isfile() else None
                        )
                        try:
                            target.addfile(clean, content)
                        finally:
                            if content is not None:
                                content.close()
        check_archive(Path(temporary))
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument(
        "--check", action="store_true", help="Validate metadata without rewriting"
    )
    args = parser.parse_args()
    for path in args.archives:
        if not args.check:
            sanitize_archive(path)
        check_archive(path)
        print(f"Archive metadata clean: {path.name}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class ReleaseMetadataTests(unittest.TestCase):
    def test_sanitizer_removes_ownership_and_preserves_contents(self):
        script = Path(__file__).resolve().parents[1] / "scripts/sanitize_sdist.py"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.tar.gz"
            with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
                member = tarfile.TarInfo("fixture/README.md")
                member.uid, member.gid = 1234, 5678
                member.uname, member.gname = "local-user", "local-group"
                member.mtime = 1234567890.5
                member.pax_headers = {"atime": "1234567890.5"}
                payload = b"Synthetic release fixture\n"
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            subprocess.run(
                [sys.executable, str(script), str(path)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [sys.executable, str(script), "--check", str(path)],
                check=True,
                capture_output=True,
            )
            with tarfile.open(path) as archive:
                member = archive.getmembers()[0]
                self.assertEqual(
                    (member.uid, member.gid, member.uname, member.gname, member.mtime),
                    (0, 0, "", "", 0),
                )
                self.assertEqual(archive.extractfile(member).read(), payload)
            first = path.read_bytes()
            subprocess.run(
                [sys.executable, str(script), str(path)],
                check=True,
                capture_output=True,
            )
            self.assertEqual(path.read_bytes(), first)

    def test_sanitizer_rejects_links_without_overwriting_archive(self):
        script = Path(__file__).resolve().parents[1] / "scripts/sanitize_sdist.py"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo("fixture/link")
                member.type, member.linkname = tarfile.SYMTYPE, "../../outside"
                archive.addfile(member)
            original = path.read_bytes()
            result = subprocess.run(
                [sys.executable, str(script), str(path)], capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Run the pinned Megatron CPU golden-value check on saved training prefixes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "prefix.json"
TEST = "tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py"


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    fixture = json.loads(DATA.read_text(encoding="utf-8"))
    if fixture["schema"] != 1:
        raise ValueError("Unsupported prefix fixture schema")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = []
    with tempfile.TemporaryDirectory(prefix="fastci-megatron-prefix-") as directory:
        source = Path(directory)
        # Only the upstream Python validation helpers and their test configs are needed.
        paths = ["tests/__init__.py", "tests/functional_tests/python_test_utils"]
        for row in fixture["cases"]:
            model, case = row["test"].split("/")
            paths.append(
                f"tests/functional_tests/test_cases/{model}/{case}/model_config.yaml"
            )
        with tempfile.TemporaryFile() as archive:
            subprocess.run(
                ["git", "-C", str(args.repo), "archive", fixture["revision"], *paths],
                stdout=archive,
                check=True,
            )
            archive.seek(0)
            subprocess.run(
                ["tar", "-xf", "-", "-C", directory], stdin=archive, check=True
            )
        for row, config_path in zip(fixture["cases"], paths[2:], strict=True):
            case = row["test"].split("/")[1]
            destination = out / case
            destination.mkdir(exist_ok=True)
            golden, observed = (
                destination / "golden.json",
                destination / "observed.json",
            )
            write(golden, row["golden"])
            write(observed, row["observed"])
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                TEST,
                "--golden-values-path",
                str(golden),
                "--actual-values-path",
                str(observed),
                "--train-iters",
                str(row["steps"]),
                "--model-config-path",
                str(source / config_path),
            ]
            env = {
                **os.environ,
                "PYTHONPATH": str(source),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
            log = destination / "pytest.log"
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command,
                    cwd=source,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if process.returncode:
                parser.exit(1, f"Megatron golden-value check failed. See {log}\n")
            report.append(
                {"test": row["test"], "steps": row["steps"], "status": "passed"}
            )
    result = out / "prefix-verification.json"
    write(result, report)
    print(result)


if __name__ == "__main__":
    main()

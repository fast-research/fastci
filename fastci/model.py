from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Atomic replacement; CLI state commands are single-writer operations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class TestSpec:
    name: str
    command: list[str] = field(default_factory=list)
    gpus: int = 0
    hardware: str = "cpu"
    seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] = field(default_factory=dict)
    # Dimension -> how to pass an override to the existing runner. "env" or "arg".
    bindings: dict[str, dict[str, str]] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 3600

    def __post_init__(self) -> None:
        if not self.name or type(self.gpus) is not int or self.gpus < 0:
            raise ValueError("Test name and nonnegative integer GPU count are required")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if self.seconds is not None and (
            not math.isfinite(self.seconds) or self.seconds < 0
        ):
            raise ValueError("seconds must be finite and nonnegative")
        if not isinstance(self.command, list) or not all(
            isinstance(x, str) for x in self.command
        ):
            raise ValueError("command must be an argv list, never a shell string")
        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.env.items()
        ):
            raise ValueError("env must map strings to strings")
        if "CUDA_VISIBLE_DEVICES" in self.env:
            raise ValueError("GPU visibility is owned by the resource pool")

    def validation_key(self) -> str:
        dimensions = set(self.metadata.get("validation_dimensions", []))
        unknown = (
            not self.metadata.get("task_type")
            or self.metadata.get("resources_unresolved")
            or any(dimension not in self.workload for dimension in dimensions)
        )
        if unknown:
            return json.dumps({"unknown": self.name}, sort_keys=True)
        # Dimension labels alone do not encode validation semantics: sequence
        # 128 and sequence 8192 must not collapse into one representative.
        relevant = {
            key: value
            for key, value in self.workload.items()
            if key in dimensions
            or key not in self.metadata.get("unrelated_dimensions", [])
        }
        return json.dumps(
            [self.metadata, relevant, self.hardware, self.gpus], sort_keys=True
        )

    def cost(
        self, traced_seconds: float | None = None, measured: bool = False
    ) -> float:
        if self.metadata.get("resources_unresolved"):
            return math.inf
        seconds = (
            traced_seconds if measured and traced_seconds is not None else self.seconds
        )
        if seconds is None:
            seconds = traced_seconds
        return self.gpus * seconds if seconds is not None else math.inf


def load_suite(path: str | Path) -> dict[str, TestSpec]:
    return suite_from_json(read_json(path))


def suite_from_json(data: dict) -> dict[str, TestSpec]:
    tests = [TestSpec(**row) for row in data["tests"]]
    result = {test.name: test for test in tests}
    if len(result) != len(tests):
        raise ValueError("Duplicate test names")
    return result


def suite_json(tests: dict[str, TestSpec]) -> dict:
    return {"schema": 1, "tests": [asdict(t) for t in tests.values()]}

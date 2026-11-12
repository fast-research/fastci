"""Execute actual CPU assertions and export observed calls as Speedscope.

This deterministic call profiler substitutes for py-spy in the mock scenario.
Two named entities are deliberately omitted to exercise missing-evidence paths.
It is not a statistical sampler and cannot measure sampling recall or overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from workloads import TESTS

OMITTED = {
    "framework.model.Encoder.validate_config",
    "framework.ops.unobserved_activation",
}


class CallCapture:
    def __init__(self, root):
        self.root = root
        self.frames, self.frame_ids, self.samples = [], {}, []

    def frame(self, frame):
        try:
            path = Path(frame.f_code.co_filename).resolve().relative_to(self.root)
        except ValueError:
            return None
        name = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        if not hasattr(frame.f_code, "co_qualname") and "self" in frame.f_locals:
            name = type(frame.f_locals["self"]).__qualname__ + "." + name
        entity = path.as_posix().removesuffix(".py").replace("/", ".") + "." + name
        if entity in OMITTED:
            return None
        key = (str(path), name, frame.f_code.co_firstlineno)
        if key not in self.frame_ids:
            self.frame_ids[key] = len(self.frames)
            self.frames.append(
                {
                    "file": "/mock/source/" + path.as_posix(),
                    "name": name,
                    "line": frame.f_code.co_firstlineno,
                }
            )
        return self.frame_ids[key]

    def capture(self, frame, event, arg):
        if event != "call":
            return
        stack = []
        while frame:
            index = self.frame(frame)
            if index is not None:
                stack.append(index)
            frame = frame.f_back
        if stack:
            self.samples.append(list(reversed(stack)))

    def save(self, path, seconds):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "$schema": "https://www.speedscope.app/file-format-schema.json",
            "shared": {"frames": self.frames},
            "profiles": [
                {
                    "type": "sampled",
                    "name": "deterministic mock call capture",
                    "unit": "seconds",
                    "startValue": 0,
                    "endValue": seconds,
                    "samples": self.samples,
                    "weights": [seconds / len(self.samples)] * len(self.samples),
                }
            ],
            "mock_capture": {
                "kind": "Python call instrumentation, not runtime sampling",
                "deliberately_omitted": sorted(OMITTED),
            },
        }
        path.write_text(json.dumps(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=sorted(TESTS), required=True)
    args = parser.parse_args()
    layers, sequence = (
        int(os.environ.get("LAYERS", "4")),
        int(os.environ.get("SEQUENCE", "1024")),
    )
    capture = CallCapture(Path(__file__).resolve().parent)
    started = time.monotonic()
    sys.setprofile(capture.capture)
    try:
        TESTS[args.test](layers, sequence)
        # Hold mock reservations long enough to observe overlapping dispatches.
        time.sleep(float(os.environ.get("MOCK_WORK_SECONDS", ".025")))
    finally:
        sys.setprofile(None)
        path = Path(os.environ["FASTCI_TRACE_DIR"]) / f"{os.getpid()}.speedscope"
        capture.save(path, time.monotonic() - started)
        print(
            json.dumps(
                {
                    "test": args.test,
                    "layers": layers,
                    "sequence": sequence,
                    "mock_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                }
            )
        )


if __name__ == "__main__":
    main()

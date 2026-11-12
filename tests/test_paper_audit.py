from __future__ import annotations

import unittest

from fastci.graph import Evidence, Graph
from fastci.model import TestSpec
from fastci.selection import select
from fastci.source import Change, SourceIndex
from fastci.workload import invocation


class PaperAlignmentRegressionTests(unittest.TestCase):
    def test_static_expansion_includes_assignment_aliases_and_callback_references(self):
        index = SourceIndex(
            {
                "m.py": "def f(): pass\nalias = f\nsecond = alias\n"
                "def direct(): f()\ndef indirect(): second()\n"
                "def callback(): return invoke(f)\n"
            }
        )
        self.assertTrue(
            {"m.direct", "m.indirect", "m.callback"}.issubset(index.expand("m.f"))
        )

    def test_equal_dimension_names_do_not_hide_different_validation_values(self):
        meta = {"task_type": "sequence", "validation_dimensions": ["sequence"]}
        suite = {
            n: TestSpec(n, gpus=1, metadata=meta, workload={"sequence": value})
            for n, value in (("short", 128), ("long", 8192))
        }
        graph = Graph({n: Evidence({"m.f"}) for n in suite})
        result = select(
            Change("b", "h", ["m.py"], ["m.f"]), graph, suite, (), list(suite)
        )
        self.assertEqual(result.retained, ["short", "long"])

    def test_unknown_hardware_and_validation_values_cannot_prove_equivalence(self):
        for meta in (
            {"task_type": "train", "resources_unresolved": True},
            {"task_type": "train", "validation_dimensions": ["parallelism"]},
        ):
            with self.subTest(meta=meta):
                suite = {n: TestSpec(n, metadata=meta) for n in ("a", "b")}
                graph = Graph({n: Evidence({"m.f"}) for n in suite})
                result = select(
                    Change("b", "h", ["m.py"], ["m.f"]), graph, suite, (), list(suite)
                )
                self.assertEqual(result.retained, ["a", "b"])

    def test_override_replaces_every_occurrence_including_mixed_flag_styles(self):
        test = TestSpec(
            "flags",
            command=["worker", "--layers", "4", "--layers=8", "--layers", "16"],
            bindings={"layers": {"kind": "arg", "key": "--layers"}},
        )
        command, _ = invocation(test, {"layers": 2})
        self.assertEqual(
            command, ["worker", "--layers", "2", "--layers=2", "--layers", "2"]
        )

    def test_property_getter_and_setter_both_resolve_to_class_entity(self):
        source = "class C:\n @property\n def value(self):\n  return 1\n @value.setter\n def value(self, x):\n  pass\n"
        index = SourceIndex({"m.py": source})
        self.assertEqual(index.at_line("m.py", 4).key, "m.C.value")
        self.assertEqual(index.at_line("m.py", 7).key, "m.C.value")


if __name__ == "__main__":
    unittest.main()

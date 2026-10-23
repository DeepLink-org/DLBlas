# Copyright (c) 2026, DeepLink.
"""CPU-only loader regressions; runnable without PyTorch or Triton installed.

Run with: python -m unittest discover -s tests -p test_ks_auto_bench_loader.py
Only dependency imports are stubbed; the actual evaluator loader is exercised.
"""

import importlib.util
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Constexpr:
    def __init__(self, value):
        self.value = value


class TestKsModuleLoader(unittest.TestCase):
    def setUp(self):
        triton = types.ModuleType("triton")
        language = types.ModuleType("triton.language")
        language.constexpr = Constexpr
        triton.language = language
        triton.jit = lambda function: function
        self.modules = patch.dict(
            sys.modules,
            {
                "torch": types.ModuleType("torch"),
                "triton": triton,
                "triton.language": language,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        path = (
            Path(__file__).resolve().parents[1] / "benchmarks" / "ks" / "auto_bench.py"
        )
        spec = importlib.util.spec_from_file_location("_ks_loader_test", path)
        self.bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.bench
        spec.loader.exec_module(self.bench)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def load(self, source):
        path = Path(self.directory.name) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        return self.bench.load_ks_module(path)

    def test_original_constexpr_globals_are_available_to_kernel(self):
        module = self.load(
            """
import triton
import triton.language as tl
LOG2_E = tl.constexpr(1.4426950408889634)
NEG_INF = tl.constexpr(float("-inf"))
@triton.jit
def kernel():
    return LOG2_E.value, NEG_INF.value
"""
        )
        log2_e, neg_inf = module.kernel()
        self.assertEqual(log2_e, 1.4426950408889634)
        self.assertEqual(neg_inf, -math.inf)

    def test_aliases_annotations_and_arbitrary_constant_names(self):
        imports = (
            ("import triton.language as lang", "lang.constexpr"),
            ("import triton.language", "triton.language.constexpr"),
            ("import triton as tr", "tr.language.constexpr"),
            ("from triton import language as lang", "lang.constexpr"),
            ("from triton.language import constexpr as const", "const"),
        )
        for statement, constructor in imports:
            with self.subTest(statement=statement):
                module = self.load(
                    f"""
{statement}
ARBITRARY_NAME = {constructor}(2.5)
ANNOTATED: int = {constructor}(-3)
"""
                )
                self.assertEqual(module.ARBITRARY_NAME.value, 2.5)
                self.assertEqual(module.ANNOTATED.value, -3)

    def test_plain_literals_and_source_order_are_preserved(self):
        module = self.load(
            """
import triton.language as tl
VALUE = tl.constexpr(1)
VALUE = tl.constexpr(2)
PLAIN = {"shape": (4, -8), "enabled": True}
class Model:
    value = VALUE.value
"""
        )
        self.assertEqual(module.Model.value, 2)
        self.assertEqual(module.PLAIN, {"shape": (4, -8), "enabled": True})

    def test_arbitrary_calls_and_main_body_still_do_not_execute(self):
        module = self.load(
            """
import triton.language as tl
def fail():
    raise AssertionError("must not execute")
CALL = fail()
WRAPPED_CALL = tl.constexpr(fail())
NESTED_CALL = tl.constexpr(float(fail()))
KEYWORD_CALL = tl.constexpr(value=fail())
WRONG_ARITY = tl.constexpr(1, fail())
ANNOTATED: int = fail()
if __name__ == "__main__":
    fail()
fail()
"""
        )
        for name in (
            "CALL",
            "WRAPPED_CALL",
            "NESTED_CALL",
            "KEYWORD_CALL",
            "WRONG_ARITY",
            "ANNOTATED",
        ):
            self.assertFalse(hasattr(module, name), name)

    def test_unrelated_constexpr_method_is_not_allowed(self):
        module = self.load(
            """
class Other:
    @staticmethod
    def constexpr(value):
        raise AssertionError("not Triton")
VALUE = Other.constexpr(1)
"""
        )
        self.assertFalse(hasattr(module, "VALUE"))

    def test_output_type_comparison_remains_strict(self):
        class Tensor:
            pass

        with patch.object(self.bench.torch, "Tensor", Tensor, create=True):
            with self.assertRaisesRegex(self.bench.KsCompareError, "list vs Tensor"):
                self.bench.compare_values([Tensor()], Tensor(), "output", 0.01, 0.01)


if __name__ == "__main__":
    unittest.main()

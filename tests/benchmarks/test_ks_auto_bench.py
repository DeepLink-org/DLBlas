import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch


AUTO_BENCH_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "ks" / "auto_bench.py"
)
SPEC = importlib.util.spec_from_file_location("ks_auto_bench", AUTO_BENCH_PATH)
auto_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto_bench)


class TrackingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return super().to(device)


class AutoBenchDeviceTests(unittest.TestCase):
    def test_move_model_to_device_moves_parameters(self):
        model = TrackingModel()

        result = auto_bench._move_model_to_device(model, torch.device("cpu"), "v0")

        self.assertIs(result, model)
        self.assertEqual(model.moves, [torch.device("cpu")])
        self.assertEqual(model.weight.device.type, "cpu")

    def test_move_model_to_device_wraps_failure(self):
        class BrokenModel(torch.nn.Module):
            def to(self, device):
                raise RuntimeError("move failed")

        with self.assertRaisesRegex(auto_bench.KsCompareError, "v1.*cpu.*move failed"):
            auto_bench._move_model_to_device(
                BrokenModel(), torch.device("cpu"), "case: v1"
            )

    def test_build_case_reseeds_parameterized_models(self):
        source = textwrap.dedent(
            """
            import torch

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.randn(4))

                def forward(self):
                    return self.weight

            class ModelNew(Model):
                pass

            def get_init_inputs():
                return []

            def get_inputs():
                return []
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            v0_path = Path(tmpdir) / "v0.py"
            v1_path = Path(tmpdir) / "v1.py"
            v0_path.write_text(source)
            v1_path.write_text(source)
            model, model_new, _, _ = auto_bench.build_case(v0_path, v1_path, 42)

        self.assertTrue(torch.equal(model.weight, model_new.weight))


if __name__ == "__main__":
    unittest.main()

import importlib.util
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


if __name__ == "__main__":
    unittest.main()

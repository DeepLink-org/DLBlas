import argparse
import ast
import importlib.util
import statistics
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

import torch


class KsCompareError(Exception):
    pass


@dataclass
class CaseResult:
    name: str
    passed: bool
    v0_ms: float | None = None
    v1_ms: float | None = None
    speedup: float | None = None
    message: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare KS competition v0/v1 Python files. The v0 file must define "
            "Model/get_init_inputs/get_inputs, and the v1 file must define "
            "ModelNew/get_init_inputs/get_inputs. All tensors and models must be on the same device! "
            "For example: python benchmarks/ks/auto_bench.py --v0_file dlblas/kernels/ks_competition/torch/layer_norm.py "
            "--v1_file dlblas/kernels/ks_competition/triton/layer_norm.py "
        )
    )
    parser.add_argument("--v0_file", type=Path, help="Path to the v0 .py file.")
    parser.add_argument("--v1_file", type=Path, help="Path to the v1 .py file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed case.",
    )
    parser.add_argument(
        "--full-traceback",
        action="store_true",
        help="Print full Python traceback for load/run failures.",
    )
    return parser.parse_args()


def _is_safe_literal(node):
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_safe_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _is_safe_literal(key)) and _is_safe_literal(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_safe_literal(node.operand)
    return False


def _filter_module_ast(tree):
    kept_nodes = []
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            kept_nodes.append(node)
        elif isinstance(node, ast.Assign) and _is_safe_literal(node.value):
            kept_nodes.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_safe_literal(node.value)
        ):
            kept_nodes.append(node)
    tree.body = kept_nodes
    ast.fix_missing_locations(tree)
    return tree


def load_ks_module(path: Path) -> types.ModuleType:
    if not path.exists():
        raise KsCompareError(f"file does not exist: {path}")
    if path.suffix != ".py":
        raise KsCompareError(f"expected a .py file, got: {path}")

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text()
    except OSError as exc:
        raise KsCompareError(f"failed to read {path}: {exc}") from exc

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise KsCompareError(f"syntax error in {path}:{exc.lineno}: {exc.msg}") from exc

    module_name = f"_ks_compare_{path.stem}_{abs(hash(path.resolve()))}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
    sys.modules[module_name] = module
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        code = compile(_filter_module_ast(tree), filename=str(path), mode="exec")
        exec(code, module.__dict__)
    except Exception as exc:
        raise KsCompareError(f"failed to load definitions from {path}: {exc}") from exc
    finally:
        sys.path[:] = old_sys_path
        sys.modules.pop(module_name, None)
    return module


def require_attr(module, attr_name, path: Path):
    if not hasattr(module, attr_name):
        raise KsCompareError(f"{path} must define `{attr_name}`.")
    return getattr(module, attr_name)


def call_with_context(func, description):
    try:
        return func()
    except Exception as exc:
        raise KsCompareError(f"{description} failed: {exc}") from exc


def as_args(value, description):
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise KsCompareError(
        f"{description} must return a list or tuple, got {type(value).__name__}."
    )


def set_seed(seed):
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                torch.npu.manual_seed_all(seed)
        except Exception:
            pass


def sync_devices():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                torch.npu.synchronize()
        except Exception:
            pass


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def describe_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )
    if isinstance(value, (list, tuple)):
        inner = ", ".join(describe_value(item) for item in value)
        return f"{type(value).__name__}({inner})"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{key}: {describe_value(item)}" for key, item in value.items()
        )
        return f"dict({inner})"
    return repr(value)


def compare_values(v0, v1, path, atol, rtol):
    if isinstance(v0, torch.Tensor) or isinstance(v1, torch.Tensor):
        if not isinstance(v0, torch.Tensor) or not isinstance(v1, torch.Tensor):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if v0.shape != v1.shape:
            raise KsCompareError(
                f"{path}: tensor shape mismatch: {v0.shape} vs {v1.shape}"
            )
        if (
            v0.dtype.is_floating_point
            or v1.dtype.is_floating_point
            or v0.is_complex()
            or v1.is_complex()
        ):
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            if not torch.allclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True):
                if lhs.is_complex() or rhs.is_complex():
                    diff = (lhs - rhs).abs()
                else:
                    diff = (lhs.float() - rhs.float()).abs()
                if diff.numel() == 0:
                    diff_summary = "empty tensor"
                else:
                    diff_summary = (
                        f"max_abs_diff={diff.max().item():.6e}, "
                        f"mean_abs_diff={diff.mean().item():.6e}"
                    )
                raise KsCompareError(
                    f"{path}: tensor values differ; {diff_summary}, atol={atol}, rtol={rtol}, "
                    f"v0={describe_value(v0)}, v1={describe_value(v1)}"
                )
        else:
            lhs = v0.detach()
            rhs = v1.detach().to(lhs.device)
            if not torch.equal(lhs, rhs):
                mismatch = (lhs != rhs).sum().item()
                raise KsCompareError(
                    f"{path}: tensor values differ; mismatched_elements={mismatch}, "
                    f"v0={describe_value(v0)}, v1={describe_value(v1)}"
                )
        return

    if isinstance(v0, tuple) or isinstance(v1, tuple):
        if not isinstance(v0, tuple) or not isinstance(v1, tuple):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if len(v0) != len(v1):
            raise KsCompareError(
                f"{path}: tuple length mismatch: {len(v0)} vs {len(v1)}"
            )
        for i, (item0, item1) in enumerate(zip(v0, v1)):
            compare_values(item0, item1, f"{path}[{i}]", atol, rtol)
        return

    if isinstance(v0, list) or isinstance(v1, list):
        if not isinstance(v0, list) or not isinstance(v1, list):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if len(v0) != len(v1):
            raise KsCompareError(
                f"{path}: list length mismatch: {len(v0)} vs {len(v1)}"
            )
        for i, (item0, item1) in enumerate(zip(v0, v1)):
            compare_values(item0, item1, f"{path}[{i}]", atol, rtol)
        return

    if isinstance(v0, dict) or isinstance(v1, dict):
        if not isinstance(v0, dict) or not isinstance(v1, dict):
            raise KsCompareError(
                f"{path}: output type mismatch: {type(v0).__name__} vs {type(v1).__name__}"
            )
        if set(v0) != set(v1):
            raise KsCompareError(
                f"{path}: dict keys mismatch: {sorted(v0)} vs {sorted(v1)}"
            )
        for key in sorted(v0):
            compare_values(v0[key], v1[key], f"{path}[{key!r}]", atol, rtol)
        return

    if v0 != v1:
        raise KsCompareError(f"{path}: values differ: {v0!r} vs {v1!r}")


def build_case(v0_path: Path, v1_path: Path, seed: int):
    v0_module = load_ks_module(v0_path)
    v1_module = load_ks_module(v1_path)

    model_cls = require_attr(v0_module, "Model", v0_path)
    model_new_cls = require_attr(v1_module, "ModelNew", v1_path)
    v0_get_init_inputs = require_attr(v0_module, "get_init_inputs", v0_path)
    v1_get_init_inputs = require_attr(v1_module, "get_init_inputs", v1_path)
    v0_get_inputs = require_attr(v0_module, "get_inputs", v0_path)
    v1_get_inputs = require_attr(v1_module, "get_inputs", v1_path)

    for func, name, path in (
        (v0_get_init_inputs, "get_init_inputs", v0_path),
        (v1_get_init_inputs, "get_init_inputs", v1_path),
        (v0_get_inputs, "get_inputs", v0_path),
        (v1_get_inputs, "get_inputs", v1_path),
    ):
        if not callable(func):
            raise KsCompareError(f"{path}: `{name}` must be callable.")

    set_seed(seed)
    v0_init_args = as_args(
        call_with_context(v0_get_init_inputs, f"{v0_path}: get_init_inputs()"),
        f"{v0_path}: get_init_inputs()",
    )
    set_seed(seed)
    v1_init_args = as_args(
        call_with_context(v1_get_init_inputs, f"{v1_path}: get_init_inputs()"),
        f"{v1_path}: get_init_inputs()",
    )

    model = call_with_context(
        lambda: model_cls(*v0_init_args), f"{v0_path}: Model(...)"
    )
    model_new = call_with_context(
        lambda: model_new_cls(*v1_init_args), f"{v1_path}: ModelNew(...)"
    )
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model_new, "eval"):
        model_new.eval()

    set_seed(seed)
    v0_inputs = as_args(
        call_with_context(v0_get_inputs, f"{v0_path}: get_inputs()"),
        f"{v0_path}: get_inputs()",
    )
    set_seed(seed)
    v1_inputs = as_args(
        call_with_context(v1_get_inputs, f"{v1_path}: get_inputs()"),
        f"{v1_path}: get_inputs()",
    )

    if len(v0_inputs) != len(v1_inputs):
        raise KsCompareError(
            f"get_inputs argument count mismatch: {v0_path} returns {len(v0_inputs)} "
            f"args, {v1_path} returns {len(v1_inputs)} args."
        )
    return model, model_new, v0_inputs, v1_inputs


def run_forward(model, inputs, seed, description):
    set_seed(seed)
    cloned_inputs = clone_value(inputs)
    try:
        with torch.no_grad():
            return model.forward(*cloned_inputs)
    except Exception as exc:
        raise KsCompareError(f"{description} forward failed: {exc}") from exc


def time_forward(model, inputs, seed, warmup, repeat):
    def one_call():
        with torch.no_grad():
            model.forward(*inputs)

    for _ in range(warmup):
        one_call()
    sync_devices()

    samples = []
    for _ in range(repeat):
        set_seed(seed)
        start = time.perf_counter()
        one_call()
        sync_devices()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _get_model_device(model):
    """Return the device of *model*'s first parameter or buffer, or None."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        pass
    return None


def _first_input_device(inputs):
    """Return the device of the first tensor found in nested *inputs*, or None."""
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            d = _first_input_device(item)
            if d is not None:
                return d
    if isinstance(inputs, dict):
        for item in inputs.values():
            d = _first_input_device(item)
            if d is not None:
                return d
    return None


def _detect_target_device(model, model_new, v0_inputs, v1_inputs):
    """Pick a non-CPU device from models/inputs, or auto-detect one.

    Priority: model device > input device > auto-detect (cuda → npu).
    Raises KsCompareError if no accelerator is available.
    """
    for m in (model, model_new):
        d = _get_model_device(m)
        if d is not None and d.type != "cpu":
            return d
    for inputs in (v0_inputs, v1_inputs):
        d = _first_input_device(inputs)
        if d is not None and d.type != "cpu":
            return d
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return torch.device("npu")
        except Exception:
            pass
    raise KsCompareError(
        "no accelerator device available (cuda/npu); "
        "cannot run accuracy or performance comparison on CPU."
    )


def _move_to_device(value, device):
    """Recursively copy every tensor in *value* to *device*."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def compare_case(name, v0_path, v1_path, args):
    model, model_new, v0_inputs, v1_inputs = build_case(v0_path, v1_path, args.seed)

    target_device = _detect_target_device(model, model_new, v0_inputs, v1_inputs)
    v0_inputs = _move_to_device(v0_inputs, target_device)
    v1_inputs = _move_to_device(v1_inputs, target_device)

    v0_output = run_forward(model, v0_inputs, args.seed, f"{name}: v0")
    v1_output = run_forward(model_new, v1_inputs, args.seed, f"{name}: v1")
    compare_values(v0_output, v1_output, "output", args.atol, args.rtol)

    v0_ms = time_forward(model, v0_inputs, args.seed, args.warmup, args.repeat)
    v1_ms = time_forward(model_new, v1_inputs, args.seed, args.warmup, args.repeat)
    speedup = v0_ms / v1_ms if v1_ms > 0 else float("inf")
    return CaseResult(name=name, passed=True, v0_ms=v0_ms, v1_ms=v1_ms, speedup=speedup)


def main():
    args = parse_args()
    v0_path = args.v0_file.resolve()
    v1_path = args.v1_file.resolve()
    if not v0_path.is_file():
        raise SystemExit(f"v0_file is not a file: {v0_path}")
    if not v1_path.is_file():
        raise SystemExit(f"v1_file is not a file: {v1_path}")
    if v0_path.suffix != ".py":
        raise SystemExit(f"v0_file must be a .py file: {v0_path}")
    if v1_path.suffix != ".py":
        raise SystemExit(f"v1_file must be a .py file: {v1_path}")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0.")

    name = str(v0_path)

    try:
        result = compare_case(name, v0_path, v1_path, args)
        print(
            f"PASS accuracy; v0={result.v0_ms:.6f} ms, "
            f"v1={result.v1_ms:.6f} ms, speedup={result.speedup:.3f}x"
        )
        passed = 1
        failed = 0
    except Exception as exc:
        if args.full_traceback:
            traceback.print_exc()
        message = str(exc)
        result = CaseResult(name=name, passed=False, message=message)
        print(f"FAIL {message}")
        passed = 0
        failed = 1

    print(f"\nSummary: {passed} passed, {failed} failed, 1 total.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

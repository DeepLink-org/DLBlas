import functools
import importlib
import operator
import sys
from typing import Any, Callable, Dict, Optional, Union

import torch
import triton
from packaging.version import Version

from dlblas.utils.device_utils import infer_device, is_npu


def get_tl_exp():
    if is_npu():
        from triton.language.math import exp as tl_exp
    elif triton.__version__ >= "3.0.0":
        from triton.language.extra.cuda.libdevice import fast_expf as tl_exp
    else:
        from triton.language.math import fast_expf as tl_exp
    return tl_exp


exp = get_tl_exp()


def get_tl_log():
    if is_npu():
        from triton.language.math import log as tl_log
    elif triton.__version__ >= "3.0.0":
        from triton.language.extra.cuda.libdevice import fast_logf as tl_log
    else:
        from triton.language.math import fast_logf as tl_log
    return tl_log


def get_tl_tanh():
    if is_npu():
        try:
            from triton.language.extra.ascend.libdevice import tanh
        except ModuleNotFoundError:
            tanh = None
    if triton.__version__ >= "3.0.0":
        try:
            # typical import path with dispatch available
            from triton.language.extra.libdevice import tanh
        except ModuleNotFoundError:
            # for working with NGC containers
            from triton.language.extra.cuda.libdevice import tanh
    else:
        from triton.language.math import tanh
    return tanh


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _is_equal(a, b):
    if isinstance(a, torch.Tensor):
        return a is b
    # Whitelist of types that are safe to compare by value for caching.
    if isinstance(a, (int, float, str, bool, type(None))) and isinstance(
        b, (int, float, str, bool, type(None))
    ):
        return a == b
    # For other types, we cannot guarantee a cheap and safe comparison, so we fail the cache check.
    return False


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                # For Tensors, check for object identity. For other types, check for equality.
                # Python caches small integers, so `is` works for them but not for large integers like 4096.
                if (
                    all(_is_equal(a, b) for a, b in zip(args, last_args))
                    and set(kwargs.keys()) == set(last_kwargs.keys())
                    and all(_is_equal(v, last_kwargs[k]) for k, v in kwargs.items())
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def calculate_tensor_similarity(x, y, name="tensor"):
    """
    Calculate similarity between two tensors using a normalized dot product metric.

    Unlike torch.testing.assert_close which uses absolute/relative tolerance based on
    element-wise differences, this function computes a global similarity score:
        sim = 2 * <x, y> / (||x||^2 + ||y||^2)

    This metric is scale-invariant and measures the cosine-like similarity normalized
    by the magnitude of both tensors. It returns 1 for identical tensors and values
    closer to 0 for dissimilar ones. This is particularly useful for comparing tensors
    with varying magnitudes where relative errors matter more than absolute differences.

    Args:
        x: First tensor to compare
        y: Second tensor to compare
        name: Name of the tensor for logging purposes

    Returns:
        Similarity score in range [0, 1] where 1 means identical
    """
    x, y = x.data.double(), y.data.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        print(f"\033[33mWARNING: {name} all zero\033[0m")
        return 1
    sim = 2 * (x * y).sum() / denominator
    return sim


def assert_tensors_similar(x, y, eps=1e-8, name="tensor", raise_assert=True):
    """
    Assert that two tensors are similar using a global similarity metric.

    Key differences from torch.testing.assert_close:
    - torch.testing.assert_close: Uses element-wise comparison with rtol/atol, checking
      that |x - y| <= atol + rtol * |y| for each element. It's sensitive to outliers
      and requires all elements to satisfy the tolerance.
    - assert_tensors_similar: Uses a single global similarity score (1 - sim) where sim is the
      normalized dot product. It's more robust to outliers and focuses on overall
      tensor similarity rather than element-wise precision. This is better suited for
      comparing large tensors where a few outlier elements shouldn't fail the test.

    Args:
        x: First tensor to compare
        y: Second tensor to compare
        eps: Maximum allowed difference (1 - similarity), default 1e-8
        name: Name of the tensor for error messages
        raise_assert: Whether to raise assertion error on failure
    """
    sim = calculate_tensor_similarity(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print(
            f"\033[31mERROR: {name} similarity check failed, diff={diff:.2e} (threshold={eps:.2e})\033[0m"
        )
        if raise_assert:
            assert False  # noqa: B011


class DisposibleTensor:

    def __init__(self, value: torch.Tensor):
        self._value = value
        self._backup_metadata: Optional[Dict[str, Any]] = None

    @property
    def value(self):
        assert not self.is_disposed
        return self._value

    def dispose(self, backup_metadata: bool = True):
        assert not self.is_disposed

        if not torch.compiler.is_compiling():
            refcount = sys.getrefcount(self._value)
            assert refcount == 2, f"refcount={refcount}"

        if backup_metadata:
            self._backup_metadata = self._compute_backup_metadata(self._value)

        self._value = None

    @property
    def is_disposed(self):
        return self._value is None

    @staticmethod
    def maybe_unwrap(value: "MaybeDisposibleTensor") -> torch.Tensor:
        if isinstance(value, DisposibleTensor):
            return value.value
        return value

    @staticmethod
    def maybe_dispose(value: "MaybeDisposibleTensor") -> torch.Tensor:
        if isinstance(value, DisposibleTensor):
            value.dispose()

    @property
    def shape(self):
        return self._get_metadata("shape")

    @property
    def device(self):
        return self._get_metadata("device")

    @property
    def dtype(self):
        return self._get_metadata("dtype")

    def _get_metadata(self, name: str):
        if not self.is_disposed:
            return getattr(self._value, name)
        assert (
            self._backup_metadata is not None
        ), "Use backup_metadata flag if you want to use metadata after dispose"
        return self._backup_metadata[name]

    _BACKUP_METADATA_KEYS = ["shape", "device", "dtype"]

    @staticmethod
    def _compute_backup_metadata(value: torch.Tensor):
        return {k: getattr(value, k) for k in DisposibleTensor._BACKUP_METADATA_KEYS}


MaybeDisposibleTensor = Union[torch.Tensor, DisposibleTensor]


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def get_amp_custom_fwd_bwd() -> Callable:
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args, strict=False)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper

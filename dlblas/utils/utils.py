import functools
import importlib
import operator
import sys
from typing import Any, Callable, Dict, Optional, Union

import torch
from packaging.version import Version


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


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
    def maybe_unwrap(value: 'MaybeDisposibleTensor') -> torch.Tensor:
        if isinstance(value, DisposibleTensor):
            return value.value
        return value

    @staticmethod
    def maybe_dispose(value: 'MaybeDisposibleTensor') -> torch.Tensor:
        if isinstance(value, DisposibleTensor):
            value.dispose()

    @property
    def shape(self):
        return self._get_metadata('shape')

    @property
    def device(self):
        return self._get_metadata('device')

    @property
    def dtype(self):
        return self._get_metadata('dtype')

    def _get_metadata(self, name: str):
        if not self.is_disposed:
            return getattr(self._value, name)
        assert (self._backup_metadata is not None), 'Use backup_metadata flag if you want to use metadata after dispose'
        return self._backup_metadata[name]

    _BACKUP_METADATA_KEYS = ['shape', 'device', 'dtype']

    @staticmethod
    def _compute_backup_metadata(value: torch.Tensor):
        return {k: getattr(value, k) for k in DisposibleTensor._BACKUP_METADATA_KEYS}


MaybeDisposibleTensor = Union[torch.Tensor, DisposibleTensor]


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def get_amp_custom_fwd_bwd() -> Callable:
    device = infer_device()
    if compare_version('torch', operator.ge, '2.4.0'):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()

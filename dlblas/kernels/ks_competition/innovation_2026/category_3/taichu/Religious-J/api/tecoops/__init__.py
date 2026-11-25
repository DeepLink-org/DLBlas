import glob
import os


def _parse_cpu_list(cpu_list):
    cpus = set()
    for item in cpu_list.strip().split(","):
        if not item:
            continue
        bounds = item.split("-", 1)
        first = int(bounds[0])
        last = int(bounds[-1])
        cpus.update(range(first, last + 1))
    return cpus


def _bind_host_thread_to_sdaa_numa():
    """Keep Host launches on the CPU node attached to the SDAA device."""
    if not hasattr(os, "sched_setaffinity"):
        return
    if os.environ.get("TECOOPS_DISABLE_NUMA_AFFINITY") == "1":
        return

    local_cpu_sets = set()
    for path in glob.iglob(
        "/sys/class/aicard/tcaicard*/device/local_cpulist"
    ):
        try:
            with open(path, encoding="ascii") as cpu_list_file:
                local_cpus = _parse_cpu_list(cpu_list_file.read())
        except (OSError, ValueError):
            continue
        if local_cpus:
            local_cpu_sets.add(frozenset(local_cpus))

    # A single CPU set is unambiguous. On a multi-NUMA/multi-card host, keep
    # the caller's policy unless device-to-card mapping is available.
    if len(local_cpu_sets) != 1:
        return

    try:
        allowed_cpus = os.sched_getaffinity(0)
        local_cpus = allowed_cpus.intersection(next(iter(local_cpu_sets)))
        if local_cpus and local_cpus != allowed_cpus:
            os.sched_setaffinity(0, local_cpus)
    except OSError:
        # Sandboxed runtimes may expose topology while denying affinity changes.
        return


_bind_host_thread_to_sdaa_numa()

del _bind_host_thread_to_sdaa_numa
del _parse_cpu_list
del glob
del os

import torch

from ._torch_ext import flatten_rays, morton3D_invert, reshape_and_cache, rms_norm, flash_attn_varlen_func, causal_conv1d_fn_torch

__all__ = ['flatten_rays', 'morton3D_invert', 'reshape_and_cache', 'rms_norm', 'flash_attn_varlen_func', 'causal_conv1d_fn_torch']

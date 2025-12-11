import torch

from dlblas.kernels.ascend.rms_norm import rms_norm_block_triton, rms_norm_triton
from dlblas.utils.device_utils import DEVICE
from tests.kernels.ascend.common import benchmark_test


def rms_norm_eager_ref(input: torch.Tensor, weight: torch.Tensor, eps: torch.Tensor):
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


def rms_norm_ref(input: torch.Tensor, weight: torch.Tensor, eps: float):
    return torch.rms_norm(input, normalized_shape=weight.shape, weight=weight, eps=eps)


def test_rms_norm(do_bench=False):
    dtype_ = torch.float16
    b = 32
    seq_len = 4096
    q_lora_rank = 1536  # q 低秩矩阵维度
    kv_lora_rank = 512  # Hckv: kv 低秩矩阵维度
    out_matmul_cq = torch.randn((b, seq_len, q_lora_rank), dtype=dtype_, device=DEVICE)
    value_states = torch.randn(
        (b, seq_len, 1, kv_lora_rank), dtype=dtype_, device=DEVICE
    )
    rmsnormGammaCq = torch.randn((q_lora_rank), dtype=dtype_, device=DEVICE)
    rmsnormGammaCkv = torch.randn((kv_lora_rank), dtype=dtype_, device=DEVICE)
    rmsnorm_cq_out_ref = rms_norm_eager_ref(out_matmul_cq, rmsnormGammaCq, eps=1e-06)
    rmsnorm_cq_out_triton = rms_norm_triton(out_matmul_cq, rmsnormGammaCq, eps=1e-06)
    torch.testing.assert_close(
        rmsnorm_cq_out_ref, rmsnorm_cq_out_triton, rtol=1e-02, atol=1e-02
    )
    kv_cache_ref = rms_norm_ref(value_states, rmsnormGammaCkv, eps=1e-06)
    kv_cache_triton = rms_norm_triton(value_states, rmsnormGammaCkv, eps=1e-06)
    torch.testing.assert_close(kv_cache_ref, kv_cache_triton, rtol=1e-02, atol=1e-02)

    kv_cache_ref = rms_norm_ref(value_states, rmsnormGammaCkv, eps=1e-06)
    kv_cache_triton = rms_norm_block_triton(value_states, rmsnormGammaCkv, eps=1e-06)
    torch.testing.assert_close(kv_cache_ref, kv_cache_triton, rtol=1e-02, atol=1e-02)
    if do_bench:
        benchmark_test(
            rms_norm_ref,
            rms_norm_triton,
            (value_states, rmsnormGammaCkv, 1e-06),
            "rms_norm_triton",
        )
        benchmark_test(
            rms_norm_ref,
            rms_norm_block_triton,
            (value_states, rmsnormGammaCkv, 1e-06),
            "rms_norm_block_triton",
        )


if __name__ == "__main__":
    test_rms_norm(do_bench=True)

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def _frames_transform_kernel(
    coordinate_ptr,
    frame_idx_ptr,
    output_ptr,
    N,
    M,
    stride_coord_n,
    stride_coord_c,
    stride_idx_m,
    stride_idx_c,
    stride_out_m,
    stride_out_n,
    stride_out_c,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    frame_id = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)

    # Load frame indices
    idx0 = tl.load(frame_idx_ptr + frame_id * stride_idx_m + 0 * stride_idx_c)
    idx1 = tl.load(frame_idx_ptr + frame_id * stride_idx_m + 1 * stride_idx_c)
    idx2 = tl.load(frame_idx_ptr + frame_id * stride_idx_m + 2 * stride_idx_c)

    # Load coordinates of the three frame atoms
    a0 = tl.load(coordinate_ptr + idx0 * stride_coord_n + 0 * stride_coord_c)
    a1 = tl.load(coordinate_ptr + idx0 * stride_coord_n + 1 * stride_coord_c)
    a2 = tl.load(coordinate_ptr + idx0 * stride_coord_n + 2 * stride_coord_c)

    b0 = tl.load(coordinate_ptr + idx1 * stride_coord_n + 0 * stride_coord_c)
    b1 = tl.load(coordinate_ptr + idx1 * stride_coord_n + 1 * stride_coord_c)
    b2 = tl.load(coordinate_ptr + idx1 * stride_coord_n + 2 * stride_coord_c)

    c0 = tl.load(coordinate_ptr + idx2 * stride_coord_n + 0 * stride_coord_c)
    c1 = tl.load(coordinate_ptr + idx2 * stride_coord_n + 1 * stride_coord_c)
    c2 = tl.load(coordinate_ptr + idx2 * stride_coord_n + 2 * stride_coord_c)

    # Build orthonormal basis (exactly as original)
    ab0 = a0 - b0
    ab1 = a1 - b1
    ab2 = a2 - b2
    ab_norm_raw = tl.sqrt(ab0 * ab0 + ab1 * ab1 + ab2 * ab2)
    ab_norm = ab_norm_raw + eps
    w10 = ab0 / ab_norm
    w11 = ab1 / ab_norm
    w12 = ab2 / ab_norm

    cb0 = c0 - b0
    cb1 = c1 - b1
    cb2 = c2 - b2
    cb_norm_raw = tl.sqrt(cb0 * cb0 + cb1 * cb1 + cb2 * cb2)
    cb_norm = cb_norm_raw + eps
    w20 = cb0 / cb_norm
    w21 = cb1 / cb_norm
    w22 = cb2 / cb_norm

    sum0 = w10 + w20
    sum1 = w11 + w21
    sum2 = w12 + w22
    sum_norm_raw = tl.sqrt(sum0 * sum0 + sum1 * sum1 + sum2 * sum2)
    sum_norm = sum_norm_raw + eps
    e10 = sum0 / sum_norm
    e11 = sum1 / sum_norm
    e12 = sum2 / sum_norm

    diff0 = w20 - w10
    diff1 = w21 - w11
    diff2 = w22 - w12
    diff_norm_raw = tl.sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2)
    diff_norm = diff_norm_raw + eps
    e20 = diff0 / diff_norm
    e21 = diff1 / diff_norm
    e22 = diff2 / diff_norm

    e30 = e11 * e22 - e12 * e21
    e31 = e12 * e20 - e10 * e22
    e32 = e10 * e21 - e11 * e20

    # Process atoms in chunks of BLOCK_SIZE
    for base in range(0, N, BLOCK_SIZE):
        atom_idx = base + tid
        mask = atom_idx < N

        # Load atom coordinates
        x0 = tl.load(
            coordinate_ptr + atom_idx * stride_coord_n + 0 * stride_coord_c, mask=mask
        )
        x1 = tl.load(
            coordinate_ptr + atom_idx * stride_coord_n + 1 * stride_coord_c, mask=mask
        )
        x2 = tl.load(
            coordinate_ptr + atom_idx * stride_coord_n + 2 * stride_coord_c, mask=mask
        )

        # Displacement from frame origin b
        d0 = x0 - b0
        d1 = x1 - b1
        d2 = x2 - b2

        # Project onto orthonormal basis
        out0 = d0 * e10 + d1 * e11 + d2 * e12
        out1 = d0 * e20 + d1 * e21 + d2 * e22
        out2 = d0 * e30 + d1 * e31 + d2 * e32

        # Store to output
        out_base = output_ptr + frame_id * stride_out_m + atom_idx * stride_out_n
        tl.store(out_base + 0 * stride_out_c, out0, mask=mask)
        tl.store(out_base + 1 * stride_out_c, out1, mask=mask)
        tl.store(out_base + 2 * stride_out_c, out2, mask=mask)


def frames_transform(
    coordinate: torch.Tensor, frame_atom_index: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Fused transformation using a single Triton kernel.
    """
    coordinate = coordinate.contiguous()
    frame_atom_index = frame_atom_index.contiguous()

    N = coordinate.size(0)
    M = frame_atom_index.size(0)

    output = torch.empty((M, N, 3), device=coordinate.device, dtype=coordinate.dtype)

    # Strides
    stride_coord_n = coordinate.stride(0)
    stride_coord_c = coordinate.stride(1)
    stride_idx_m = frame_atom_index.stride(0)
    stride_idx_c = frame_atom_index.stride(1)
    stride_out_m = output.stride(0)
    stride_out_n = output.stride(1)
    stride_out_c = output.stride(2)

    grid = lambda meta: (M,)
    _frames_transform_kernel[grid](
        coordinate,
        frame_atom_index,
        output,
        N,
        M,
        stride_coord_n,
        stride_coord_c,
        stride_idx_m,
        stride_idx_c,
        stride_out_m,
        stride_out_n,
        stride_out_c,
        eps,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coordinate: torch.Tensor, frame_atom_index: torch.Tensor):
        return frames_transform(coordinate, frame_atom_index)


# Hyperparameters & Data Generation
N_ATOM = 256
N_FRAME = 64


def get_inputs():
    device = "cuda"
    torch.manual_seed(42)

    coordinate = torch.randn(N_ATOM, 3, device=device)
    frame_atom_index = torch.randint(
        0, N_ATOM, (N_FRAME, 3), device=device, dtype=torch.int64
    )

    return [coordinate, frame_atom_index]


def get_init_inputs():
    return []

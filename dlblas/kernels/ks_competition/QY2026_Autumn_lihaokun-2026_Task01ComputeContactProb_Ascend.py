import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def contact_prob_kernel(
    logits_ptr,  # *f32
    out_ptr,     # *f32
    B,           # no_bins
    T,           # thres_idx
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * B
    row_ptr = logits_ptr + base

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < B

    # Safe offsets to avoid generating out-of-bounds pointers on strict backends
    safe_offsets = tl.where(mask, offsets, tl.zeros_like(offsets))
    NEG_BIG = -1.0e30
    x = tl.load(row_ptr + safe_offsets, mask=mask, other=NEG_BIG).to(tl.float32)

    # Numerically-stable softmax components
    row_max = tl.max(x, axis=0)
    # Exponentiate only valid positions and zero-out others
    exp_x = tl.where(mask, tl.exp(x - row_max), 0.0)

    # Denominator of softmax
    denom = tl.sum(exp_x, axis=0)

    # Choose to sum the smaller side and use complement if beneficial
    T_eff = tl.minimum(T, B)
    choose_left = T_eff * 2 <= B  # True -> sum bins < T; False -> sum bins >= T
    mask_left = (offsets < T_eff) & mask
    mask_right = (offsets >= T_eff) & mask
    selected_mask = tl.where(choose_left, mask_left, mask_right)

    numer = tl.sum(tl.where(selected_mask, exp_x, 0.0), axis=0)

    partial = numer / denom
    out_val = tl.where(choose_left, partial, 1.0 - partial)

    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    """
    Compute Contact Probability:
    distogram logits -> contact probability

    Input:
        distogram_logits: Tensor of shape [N_token, N_token, no_bins]

    Output:
        contact_prob: Tensor of shape [N_token, N_token]
    """

    def __init__(
        self,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        no_bins: int = 64,
        thres: float = 8.0,
    ):
        super().__init__()

        self.min_bin = float(min_bin)
        self.max_bin = float(max_bin)
        self.no_bins = int(no_bins)
        self.thres = float(thres)

        edges = torch.linspace(self.min_bin, self.max_bin, self.no_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        self.thres_idx = int((centers < self.thres).sum().item())

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distogram_logits: Tensor of shape [N_token, N_token, no_bins]

        Returns:
            contact_prob: Tensor of shape [N_token, N_token]
        """
        B = distogram_logits.size(-1)
        T = int(self.thres_idx)

        # Fast-path edge cases
        if T <= 0:
            return distogram_logits.new_zeros(distogram_logits.shape[:-1])
        if T >= B:
            return distogram_logits.new_ones(distogram_logits.shape[:-1])

        device_type = distogram_logits.device.type

        if device_type != "cuda":
            distogram_prob = torch.softmax(distogram_logits, dim=-1)
            return distogram_prob[..., :T].sum(dim=-1)

        # CUDA path: Triton kernel
        N0, N1, _ = distogram_logits.shape
        M = N0 * N1

        # Ensure [M, B] contiguous layout
        x2d = distogram_logits.reshape(M, B).contiguous()

        # Output buffer
        out = torch.empty((M,), device=distogram_logits.device, dtype=torch.float32)

        # Choose BLOCK_SIZE as next power of two >= B (capped)
        def next_pow2(x: int) -> int:
            return 1 << (x - 1).bit_length()

        BLOCK_SIZE = min(max(32, next_pow2(B)), 1024)

        # Heuristic for num_warps
        if BLOCK_SIZE <= 64:
            num_warps = 2
        elif BLOCK_SIZE <= 128:
            num_warps = 4
        else:
            num_warps = 8

        grid = (M,)
        contact_prob_kernel[grid](
            x2d, out, B, T,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )

        contact_prob = out.view(N0, N1).to(distogram_logits.dtype)
        return contact_prob


N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0


def get_inputs():
    distogram_logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS)
    return [distogram_logits]


def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]
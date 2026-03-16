import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to use Triton for fast kernels; fall back gracefully if unavailable
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: tanh-based GELU used by NewGELU (fp32 compute, cast back)
if TRITON_AVAILABLE:
    @triton.jit
    def _gelu_tanh_fwd(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        x_f32 = x.to(tl.float32)
        x2 = x_f32 * x_f32
        x3 = x2 * x_f32
        inner = x_f32 + 0.044715 * x3
        # sqrt(2/pi) * inner
        a = 0.7978845608028654 * inner
        # tanh(a) via a single exp: tanh(a) = (e^(2a) - 1) / (e^(2a) + 1)
        e2a = tl.exp(2.0 * a)
        t = (e2a - 1.0) / (e2a + 1.0)
        y = 0.5 * x_f32 * (1.0 + t)
        y = y.to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)

# Triton LayerNorm forward (affine) over last dimension
if TRITON_AVAILABLE:
    @triton.jit
    def _layer_norm_forward_kernel(
        x_ptr, w_ptr, b_ptr, y_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        x_row_ptrs = x_ptr + pid * stride_xm + offs * stride_xn
        y_row_ptrs = y_ptr + pid * stride_ym + offs * stride_yn
        w_ptrs = w_ptr + offs
        b_ptrs = b_ptr + offs

        x_in = tl.load(x_row_ptrs, mask=mask, other=0.0)
        x = x_in.to(tl.float32)

        n = tl.full([1], N, tl.float32)
        mean = tl.sum(x, axis=0) / n
        var = tl.sum(x * x, axis=0) / n - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd
        y = y * w + b
        y = y.to(x_in.dtype)

        tl.store(y_row_ptrs, y, mask=mask)

def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()

def layer_norm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    # Fallback for non-CUDA or unsupported shapes
    if (not TRITON_AVAILABLE) or (not x.is_cuda) or (x.ndim < 2):
        return torch.layer_norm(x, (x.shape[-1],), weight, bias, eps=eps)

    orig_shape = x.shape
    M = int(x.numel() // x.shape[-1])
    N = int(x.shape[-1])

    x_2d = x.view(M, N)
    y = torch.empty_like(x_2d)

    stride_xm, stride_xn = x_2d.stride()
    stride_ym, stride_yn = y.stride()

    BLOCK_N = _next_power_of_2(N)
    # Cap block size to keep resource usage reasonable
    if BLOCK_N > 4096:
        BLOCK_N = 4096
        if BLOCK_N < N:
            # Uncommon case; fallback to PyTorch for correctness
            return torch.layer_norm(x, (x.shape[-1],), weight, bias, eps=eps)

    grid = (M,)
    num_warps = 4 if BLOCK_N <= 1024 else 8
    num_stages = 2

    _layer_norm_forward_kernel[grid](
        x_2d, weight, bias, y,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.view(orig_shape)

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        # Use Triton accelerated path on CUDA; exact same tanh-based formula, computed in fp32 then cast back
        if TRITON_AVAILABLE and x.is_cuda:
            x_contig = x.contiguous()
            y = torch.empty_like(x_contig)
            n_elements = x_contig.numel()
            if n_elements > 0:
                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                _gelu_tanh_fwd[grid](x_contig, y, n_elements, BLOCK_SIZE=4096)
            return y.view_as(x)
        # CPU or fallback
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)

        # Use PyTorch's fused scaled_dot_product_attention (Flash/Math/Memory-Efficient) with causal masking.
        # This preserves semantics: same scaling (1/sqrt(hs)), same causal mask, dropout on attention weights.
        dropout_p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class ModelNew(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        # Triton-accelerated LayerNorm on CUDA; exact affine LN semantics
        ln1_out = layer_norm_triton(x, self.ln_1.weight, self.ln_1.bias, self.ln_1.eps) if (TRITON_AVAILABLE and x.is_cuda) else self.ln_1(x)
        x = x + self.attn(ln1_out)
        ln2_out = layer_norm_triton(x, self.ln_2.weight, self.ln_2.bias, self.ln_2.eps) if (TRITON_AVAILABLE and x.is_cuda) else self.ln_2(x)
        x = x + self.mlpf(ln2_out)
        return x

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
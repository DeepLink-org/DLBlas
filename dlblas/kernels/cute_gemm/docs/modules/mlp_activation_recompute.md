# MLP 激活重计算

## 概要

cute_gemm 的 `MLP` 用三种模式在内存与计算之间取舍：

| 模式 | 前向 GEMM | 反向 GEMM | 合计 | 保存的激活 | 机制 |
|---|---|---|---|---|---|
| normal | 2 | 4 | **6** | `x` + `preact` | 默认 |
| `torch.utils.checkpoint` | 2 | 2（重放）+ 4 = 6 | **8** | 仅 `x` | 包住整个 MLP |
| checkpoint + torch.compile | 2 | 2（重放）+ 4 = 6 | **8** | 仅 `x` | compile 无收益 |
| `recompute=True` | 2 | 1（重放 fc1）+ 4 = 5 | **7** | 仅 `x` | 内置，只重放 fc1 |

`recompute=True` 严格优于 `torch.utils.checkpoint`。内存节省相同，反向少一次
GEMM。checkpoint 之上加 `torch.compile` 无收益。编译器看不到自定义
`autograd.Function` 子类内部，无法知道 fc2 的重放是冗余的。反向的
`gemm_with_activation_grad` 已经从 `preact` 重算 `postact`。

## 背景

不重计算时，MLP 前向为反向保存两份张量：

- fc1（`LinearActFunc`）保存 `x`，即 fc1 的输入。它是原始输入的 view，额外
  开销约 0。
- fc2（`DActLinearFunc`）保存 `preact`，即 fc1 的预激活输出。这份是内存大头。

fc2 自身已部分重算。它存 `preact` 而不存 `postact`。反向经
`gemm_with_activation_grad` 从 `preact` 重算 `postact`。前向到反向之间多持有的
激活只有 `preact`。

`torch.utils.checkpoint` 不保存 `preact`。代价是反向重放整个前向（fc1 + fc2，
2 次 GEMM）。`recompute=True` 只重放 fc1（1 次 GEMM）。只有 `preact` 需要重算；
重放 fc2 是浪费。

## GEMM kernel 分解

实测环境：H100，bf16，形状 4×512×4096，hidden=16384。复现命令：
`scripts/mlp_checkpoint_bench.py --no-memory`。

normal（6 次 GEMM）：

```
fwd:  1× gemm_with_activation (fc1: x @ W1 + act)  +  1× gemm (fc2: postact @ W2)
bwd:  1× gemm_with_activation_grad (dout @ W2 + dact)  +  3× gemm (dW2, dx, dW1)
```

`torch.utils.checkpoint`（8 次 GEMM，加不加 `torch.compile` 相同）：

```
fwd:  1× gemm_with_activation + 1× gemm
bwd:  1× gemm_with_activation + 1× gemm (重放) + 1× gemm_with_activation_grad + 3× gemm
```

`torch.compile` 消不掉重放的那两次 GEMM。自定义 `autograd.Function` 对编译器是
不透明的图节点。

`recompute=True`（7 次 GEMM）：

```
fwd:  1× gemm_with_activation + 1× gemm
bwd:  1× gemm (只重放 fc1) + 1× gemm_with_activation_grad + 3× gemm
```

## 内存实测

实测环境：H100，bf16，输入 4×2048×4096，hidden=16384，4 层。复现命令：
`scripts/mlp_checkpoint_bench.py --no-kernels`。

```
                                    normal    checkpoint    recompute
forward activation memory          1342 MB       268 MB       268 MB
peak memory                        3221 MB      3020 MB      3020 MB
```

- 前向节省 = 层数 × preact 大小（268 MB × 4 = 1074 MB）。
- checkpoint 与 recompute 的内存曲线相同。
- recompute 每层比 checkpoint 少 1 次 GEMM。

## 用法

内置 recompute（推荐）：

```python
mlp = MLP(d_model, hidden, activation="swiglu", recompute=True)
# 或函数式：
out = mlp_func(x, w1, w2, activation="swiglu", recompute=True)
```

`torch.utils.checkpoint`（可用，每层多 1 次 GEMM）：

```python
# 在 transformer block 里：
def forward(self, x):
    x = x + torch.utils.checkpoint.checkpoint(self.mlp, x, use_reentrant=False)
    return x
```

## activation_memory_budget 对本库无效

`torch._functorch.config.activation_memory_budget` 是另一套机制：

- 即 torch.compile 的 SAC。
- 需要 `torch.compile` 追踪联合前向-反向图。
- 用 min-cut 划分器自动决定每个算子保存还是重算，预算取 0.0–1.0。
- 看不到自定义 `autograd.Function` 子类内部。cute_gemm 的 `LinearActFunc`、
  `DActLinearFunc` 对编译器不透明。

cute_gemm 的 MLP 走自定义 autograd Function。`save_for_backward` / `backward`
全部手写。`activation_memory_budget` 旋钮因此无效。请改用 `recompute=True`。

| 方案 | 需要 torch.compile | 兼容自定义 autograd.Function | 粒度 |
|---|---|---|---|
| `activation_memory_budget` | 是 | 否（不透明） | 逐算子（自动） |
| `torch.utils.checkpoint` | 否 | 是 | 逐层（手动） |
| checkpoint + torch.compile | 是 | 是，但优化不了重放 | 逐层（手动） |
| `recompute=True` | 否 | 不适用（内置） | 逐 MLP（最优） |

## 零步长梯度

`out.sum().backward()` 经 `expand()` 产生步长 `(0, 0)` 的全一梯度。GEMM kernel
要求连续步长：TMA 需要一个已知的连续维。

`cute_gemm/linear.py` 的 `_ensure_contiguous` 负责处理。以下三个类的 backward
都对 dout 调用它：

- `LinearFunc`
- `LinearActFunc`
- `DActLinearFunc`

常规路径先校验 `stride(-1) != 1`，避免多余拷贝。`torch.compile` 下无条件调用
`.contiguous()`。dynamo 无法在 fake tensor 上校验步长。

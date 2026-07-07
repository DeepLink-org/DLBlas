# engram_hash — AscendC 算子

N-gram embedding 索引哈希算子。将 N-gram token id、逐层乘子、逐表词表大小与偏移，计算为每层、每 token、每个 (ngram 位置 x embedding 表) 的嵌入索引。

## 算子规格

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|---------|------|
| ngram_token_ids | (NT, N) | int32 | N-gram token id, [0, 100000) |
| multipliers | (L, N) | int64 | 逐层哈希乘子, [0, 100000) |
| vocab_sizes | (L, N-1, T) | int32 | 词表大小, [100000, 1000000) |
| offsets | (L, W) | int32 | embedding 表偏移 (prefix-sum) |
| **输出** | (L, NT, W) | int32 | 嵌入索引 |

其中 `W = (N-1) * T`。

## 精度标准

**整数计算类：bit-exact（二进制一致 / 绝对误差 0）**。

## 构建

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd operators/engram_hash

# 构建（可执行文件 + PyTorch 扩展）
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

## 直接调用测试

```bash
# 生成测试数据 + golden
python3 scripts/gen_data.py --nt 4096 --ngram 3 --layers 2 --tables 8

# 运行（设备 2, 48 核）
ASCEND_RT_VISIBLE_DEVICES=2 ./build/engram_hash_custom 4096 3 2 8 0 48

# 精度验证（bit-exact）
python3 scripts/verify_result.py
```

## PyTorch 集成

```python
import torch
import torch_npu

torch.ops.load_library("build/libengram_hash_ops.so")
ngram = torch.randint(0, 100000, (4096, 3), dtype=torch.int32).npu()
mult  = torch.randint(0, 100000, (2, 3), dtype=torch.int64).npu()
vocab = torch.randint(100000, 1000000, (2, 2, 8), dtype=torch.int32).npu()
offs  = torch.randint(0, 1000000, (2, 16), dtype=torch.int32).npu()
out   = torch.ops.npu.engram_hash(ngram, mult, vocab, offs)
```

## 架构

- 芯片: Ascend 910B2 (DAV_2201)
- 核类型: AIV-only (48 Vector cores)
- 计算方式: 标量 int64 (mul/xor/mod/add)
- 并行策略: token 维多核切分

## 性能

| 指标 | 值 |
|------|-----|
| Geomean speedup vs PyTorch | 5.94x |
| 多核扩展 (1c→48c) | 47.78x (99.5% 效率) |
| Kernel 时间 (NT=4096, 48c) | 79 us |
| AIV scalar ratio | 98.5% |

## 文件结构

```
operators/engram_hash/
├── CMakeLists.txt                     # 双目标构建
├── op_kernel/
│   ├── engram_hash_kernel.asc         # AIV-only 标量 kernel
│   └── engram_hash_tiling.h           # Tiling 数据结构
├── op_host/
│   ├── engram_hash_host.asc           # 直接调用主程序
│   ├── engram_hash_compute_tiling.h   # Host 侧 tiling 计算
│   └── data_utils.h                   # 文件读写工具
├── op_extension/
│   ├── engram_hash_torch.cpp          # PyTorch 集成
│   ├── register.cpp                   # TORCH_LIBRARY 注册
│   └── ops.h                          # 函数声明
├── scripts/
│   ├── gen_data.py                    # 数据生成 + golden
│   ├── verify_result.py               # bit-exact 验证
│   ├── run_verify_matrix.py           # 完整验证矩阵
│   ├── test_torch.py                  # PyTorch 通路测试
│   ├── benchmark.py                   # 性能基准
│   └── build_and_test.sh              # 一键构建+测试
└── docs/
    ├── DESIGN.md                      # 架构设计
    ├── PLAN.md                        # 开发计划
    ├── environment.md                 # 环境信息
    └── perf/round_001/               # 性能采集数据
```

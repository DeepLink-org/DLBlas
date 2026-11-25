# KernelSwift算子创新大赛 启元赛道 T1-T4提交

这是同一份作品的两部分：A 部分 [ntops #105](https://github.com/InfiniTensor/ntops/pull/105) 提供算子、Torch 封装、测试和基准评测；B 部分 [ninetoothed #215](https://github.com/InfiniTensor/ninetoothed/pull/215) 提供共用的编译器和后端。

## 赛题与性能结果

| 赛题 | 评审入口 | 海光 BW / DTK 25.04 | 天数智芯天垓 150 |
| --- | --- | ---: | ---: |
| Task 01 · MXFP4 W4A16 分组专家矩阵乘 | [T1](tasks/T1/README.md) | 5.892741× | 2.970092× |
| ★ **Task 02 · Block-Scaled FP8 矩阵乘** | [T2](tasks/T2/README.md) | 3.316458× | 1.508378× |
| Task 03 · Gated RMSNorm 融合算子 | [T3](tasks/T3/README.md) | 1.2626× | 1.3436× |
| ★ **Task 04 · MLA RoPE 与压缩 KV Cache 写入融合** | [T4](tasks/T4/README.md) | 2.199483× | 1.5766× |

*★ Task 02 和 Task 04 获优胜奖。表中为各题相对各自基线的平均加速倍数，测试详情见[原始提交清单](shared/原始提交清单.md)。*

## 目录

```text
chenweihan02/
├── README.md
├── sources.lock                    # 固定源码版本，供脚本自动校验
├── tasks/
│   ├── T1/                         # 各题分别提供算子补丁、技术报告和复现说明
│   ├── T2/
│   ├── T3/
│   └── T4/
├── shared/
│   ├── ninetoothed-215.patch       # 四题共用的后端补丁
│   ├── ntops-all.patch             # 仅还原四题时使用的补充补丁
│   ├── 原始提交清单.md
│   └── 补丁验证记录.md
└── scripts/
    ├── prepare.sh                  # 按题准备源码
    └── reproduce.sh                # 在远程 GPU 环境按题评测
```

## 准备源码与复现评测

以下命令都从 `chenweihan02/` 运行。将 `/path/to/workspace` 换成已有目录；命令末尾的 `t2-source` 等子目录由脚本新建，不要提前创建。重试时请换一个新目录。

### 只准备源码

需要 Bash、Git、`awk`，并能访问 GitHub。此步骤只下载源码、应用补丁并检查版本，不安装依赖、不编译、不运行测试。

```bash
# 只看 T2；换成 T1、T3 或 T4 即可准备相应赛题
bash scripts/prepare.sh T2 /path/to/workspace/t2-source

# 一次还原四题
bash scripts/prepare.sh all /path/to/workspace/all-source
```

选单题时，只应用该题的算子改动和共用后端，不需要准备其他题。选 `all` 时，还原的源码与 `ntops@080910d` 和 `ninetoothed@6b79203` 一致，详见[补丁验证记录](shared/补丁验证记录.md)。

### 在远程 GPU 上复现评测

**仅在远程 Linux GPU 主机上运行。** 先进入对应的厂商镜像环境，再选择复现单题或全部赛题：

```bash
bash scripts/reproduce.sh --remote-gpu T2 /path/to/workspace/t2-eval
bash scripts/reproduce.sh --remote-gpu all /path/to/workspace/all-eval
```

评测脚本会自动准备源码，无需先运行 `prepare.sh`。日志写入新工作目录下的 `results/<题号>/`；环境要求和测试细节见各题的复现说明。本次只验证了补丁能否还原源码，未重新运行评测。

# T2：块缩放 FP8 GEMM

本题支持独立的行、列块缩放布局，沿 K 维分块计算并以 FP32 累加。共用后端根据目标工具链能力选择 FP8 操作数的合法化路径。

- [算子补丁](ntops.patch)：算子、Torch 封装、正确性测试、基线探测和评测脚本，源自 `ntops` 提交 `8a5be7a`。
- [技术报告](技术报告.md) · [原始复现说明](原始复现说明.md) · [共享后端补丁](../../shared/ninetoothed-215.patch)

从[交付根目录](../../README.md)执行。第一条只还原本题和共享后端源码；第二条只在远程 Linux GPU 主机执行。两条命令各使用一个新目录。

```bash
bash scripts/prepare.sh T2 /path/to/workspace/t2-source
bash scripts/reproduce.sh --remote-gpu T2 /path/to/workspace/t2-eval
```

原四题合装版本记录了 30 项算子正确性测试和 120 项定向编译器回归通过。复现入口固定使用 `vllm_portable` 兼容基线，不是原生 PyTorch 或未修改的上游 vLLM 二进制。单题布局本次仅做 Git 还原验证。

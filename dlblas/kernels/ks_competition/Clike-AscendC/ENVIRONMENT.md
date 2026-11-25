# 环境配置

该实现的构建、正确性验证和性能测试使用 HiDevLab 平台提供的 Ascend 算力
资源完成。

## 已验证环境

该实现的验证环境如下。

| 项目 | 版本或规格 |
|---|---|
| NPU | Ascend 910B1 / Atlas A2 |
| CANN | 9.0.0 |
| Python | 3.11.15，代码兼容 Python 3.10 及以上 |
| PyTorch | 2.10.0+cpu |
| torch-npu | 2.10.0 |
| CMake | 3.27.9，最低要求 3.16 |
| AscendC 架构参数 | `dav-2201` |

表中的 `torch 2.10.0+cpu` 是 torch-npu 配套 PyTorch 包的版本标记，算子
实际在 NPU 上执行。

## CANN 环境

CANN 默认安装路径为 `/usr/local/Ascend/cann-9.0.0`。使用者可在进入
DLBlas 仓库后执行：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
```

如果 CANN 安装在其他位置，`ASCEND_HOME_PATH` 需要指向对应根目录。构建
脚本从以下位置查找 AscendC CMake 工具链：

```text
${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake
```

## Python 环境检查

运行环境至少需要 `torch` 和 `torch_npu`。以下命令可用于确认版本和 NPU：

构建过程只使用 Python 解释器定位这两个包，不依赖 Python C API，因此无需
安装 `python3-dev`、`python3.10-dev` 或提供 `Python.h`。

```bash
python3 - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
PY
```

正常环境中的 `NPU available` 应为 `True`。设备状态可通过以下命令检查：

```bash
npu-smi info
```

## 可选环境变量

| 变量 | 作用 |
|---|---|
| `ASCEND_HOME_PATH` | 指定 CANN 根目录 |
| `BUILD_JOBS` | 指定 CMake 并行编译任务数，默认值为 64 |
| `DLBLAS_PYTHON_EXECUTABLE` | 指定同时安装了 `torch` 和 `torch_npu` 的 Python 解释器 |
| `DLBLAS_KS_ASCENDC_LIBRARY` | 指定已经编译好的自定义动态库绝对路径 |

构建脚本默认自动查找能够导入 `torch` 和 `torch_npu` 的解释器。存在多个
Python 环境时，可显式指定验证环境中的解释器：

```bash
DLBLAS_PYTHON_EXECUTABLE=/usr/local/python3.11.15/bin/python3 \
    bash dlblas/kernels/ks_competition/ascend/clike_910b/build.sh
```

未指定 `DLBLAS_KS_ASCENDC_LIBRARY` 时，加载器的默认查找位置为：

```text
dlblas/kernels/ks_competition/ascend/clike_910b/build/
libdlblas_ks_ascendc_ops.so
```

`build/` 为生成目录，已由 DLBlas 的 `.gitignore` 排除，不属于 PR 的提交内容。


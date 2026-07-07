# engram_hash 算子环境信息

> 此文件供 Architect / Developer / Reviewer 统一获取硬件与 CANN 环境信息。
> 内容在项目初始化时确定，随环境变更更新。

---

## 硬件

| 项目 | 值 | 来源 |
|------|:---:|------|
| **芯片型号** | Ascend 910B2 | `npu-smi info` (Product Name: IT21HMDA_Bin5) |
| **NpuArch** | DAV_2201 | `/npu-arch` skill (Ascend910B2 -> DAV_2201) |
| **`__NPU_ARCH__`** | 2201 | `/npu-arch` skill |
| **`--npu-arch` 编译参数** | dav-2201 | `/npu-arch` skill |
| **SocVersion** | ASCEND910B | `/npu-arch` skill |
| **Cube 核数** | 24 | DAV_2201 平台规格 |
| **Vector 核数** | 48 | DAV_2201 平台规格 (Cube:Vec = 1:2) |
| **UB 容量** | 192 KB (196608 B) | DAV_2201 平台规格 |
| **L1 容量** | 512 KB | DAV_2201 平台规格 |
| **L2 容量** | 192 MB | DAV_2201 平台规格 |
| **HBM Memory** | 64 GB | Ascend 910B2 典型规格 |
| **频率** | 1.8 GHz | Ascend 910B2 典型规格 |

## CANN

| 项目 | 值 | 来源 |
|------|:---:|------|
| **版本** | 9.0.0 | `/usr/local/Ascend/cann-9.0.0/version.info` |
| **发布时间** | 2026-04-28 | `version.info` timestamp |
| **安装路径** | `/usr/local/Ascend/cann-9.0.0` | 环境探测 |
| **CPU 架构** | aarch64-linux | `cann-9.0.0/` 目录结构 |

## 设备

| 项目 | 值 | 来源 |
|------|:---:|------|
| **本次使用设备** | NPU 2 | 用户指定 (health OK) |
| **固件版本** | 7.7.0.10.220 | `npu-smi info -t board -i 2` |
| **软件版本** | 25.2.3 | `npu-smi info -t board -i 2` |

## 算子路径

| 项目 | 值 |
|------|------|
| **算子名称** | engram_hash |
| **工作目录** | `/mnt/data01/zmz/workspace/12agent/waic/build/engram_hash/operators/engram_hash` |

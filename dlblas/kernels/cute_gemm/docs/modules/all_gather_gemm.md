# AllGather+GEMM

AllGather+GEMM 解决一个问题:A 沿 M 分片的多 rank GEMM,通信与计算重叠——一个持久 GEMM launch 边消费边到达的 gathered A(distributed/all_gather_gemm.py:25-29)。实现分两半:`AllGatherRunner`(all_gather_gemm.py:275)是 transport 半,持有对称缓冲、arrival flags、流与 CE 发送计划,不知道 GEMM(all_gather_gemm.py:276-279);GEMM 半是任意 cute_gemm 变体,靠一个 host 侧 `ag_args` kwarg 获得 AG 能力(all_gather_gemm.py:7-11)。模块 docstring(all_gather_gemm.py:1-249)是带测量数据的设计记录,本文只给索引与源码锚点。

约定:`路径:行号` 相对仓库根(模块内文件省略 cute_gemm/ 前缀)。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 解决什么、不是什么

| 是 | 不是 |
|---|---|
| NVLink rank 间重叠的 AG+GEMM;consumer_wait 在 kernel 内(load warp 的 flag 门,all_gather_gemm.py:128-142 映射表) | 通信库封装:内建唯一 transport 是 ce_push(CE = Copy Engine,all_gather_gemm.py:31-38),不抽象进程组以外的集合通信;两个备选 transport 已 PARKED(all_gather_gemm.py:76-105) |
| 一个 gather 供多个 GEMM 共享(q/k/v 同 gather,all_gather_gemm.py:22-23、gather docstring all_gather_gemm.py:657-658) | NCCL 替身:跨 rank 复用仍要 `dist.barrier` 与 symm_mem rendezvous(all_gather_gemm.py:350-351、:423) |
| 整次调用可外层 CUDA-graph 捕获(all_gather_gemm.py:197-230) | compile 兼容层:torch-library custom op 列为 future(all_gather_gemm.py:247) |

训练契约:backward 必须重新 gather。forward 只保存本地 shard,gathered_a():425 的视图到下一次调用即失效;runner 不区分 forward/backward,一个 (shard, dtype) 形状一个 runner 跨层共享(all_gather_gemm.py:110-118)。dgrad 的 dA 归 ReduceScatter 兄弟件,未实现(all_gather_gemm.py:122-123)。量化是调用方的事,runner 只运已量化操作数(all_gather_gemm.py:852-855)。

## 流水线词表与三条流

跨调用双缓冲生产者/消费者流水,同步对象到 full/empty 角色的映射(all_gather_gemm.py:125-142):

| 对象 | 角色 |
|---|---|
| arrival flags + epoch | 细粒度 FULL,按 (shard, chunk);consumer_wait = kernel 内 load warp 门(ag_wait_m_tile) |
| `ev_reuse[buffer]` | 粗粒度 collective EMPTY,每缓冲一个;等它 = producer_acquire(本 rank staging + 远端发送) |
| `ev_shard_staged` | producer 内部边(stage→transport),非流水 barrier |
| `ev_gemm_end` | 本 rank 的 consumer_release |
| `handle.barrier(channel=parity)` | 跨 rank consumer_release 聚合(EMPTY 是跨 rank 事实:peer 会写我的缓冲) |
| 2 个缓冲 | 流水深度,固定(all_gather_gemm.py:341-350) |

FULL 细粒度点对点(门当前消费),EMPTY 粗粒度 collective(门两次调用后的重灌)——不对称是设计而非巧合(all_gather_gemm.py:172-174)。epoch 是单调代数(MPI-RMA 意义),刻意不用 mbarrier 式 0/1 phase:i+1 拍的 FULL 信号合法地落在 i 拍消费者仍自旋期间,单调 GEQ 吸收之,parity 翻转会死锁(all_gather_gemm.py:175-178)。

三条流是实测最小集(all_gather_gemm.py:365-373):

| 流 | 跑什么 | 为什么独立 |
|---|---|---|
| ambient(调用方当前流) | epoch bump、staging、GEMM | GEMM 的 launch 不被发送的 enqueue 延迟挡住(all_gather_gemm.py:662-665) |
| push_stream | 反向环 CE 发送突发 | 必须与 GEMM 重叠 |
| barrier_stream | 仅 EMPTY 侧 barrier | 并入 push_stream 会把 sends(i+1) 排到 barrier(i) 之后,重引入跨迭代过同步;并入 ambient 则把跨 rank rendezvous 放上关键路径 |

## gather() 的一次调用

enter:`_stage_local`(:614)——等 `ev_reuse`(:628)→ bump epoch 并做 4 字节快照(:595-612)→ staging 拷贝用 SM kernel `torch.add(a, 0, out=...)` 而非 `copy_`(后者降为 CE memcpy,带 20-30 us ramp,:617-622)→ 写本地全部 chunk flags(:636-637)→ record `ev_shard_staged`。body:`yield (buf, ag_args)`(:725),consumer_wait 发生在 kernel 内。exit:record `ev_gemm_end`(:726)→ push_stream 反向环逐 peer [数据 memcpy, 4 字节 flag 远写](:741-764)→ barrier_stream 聚合 barrier 并 record reuse(:769-775)→ 捕获态下侧流 join 与 dep-restore(:776-813)。复用安全 = 2 缓冲轮转 + 一条 off 关键路径的跨 rank barrier 门住 call i+2 的 staging(:180-184)。

## ag_args 接入任意 GEMM 变体

用法(all_gather_gemm.py:14-15):

```python
with runner.gather(a_shard) as (a_full, ag_args):
    d = gemm(a_full, b, ..., ag_args=ag_args)  # 任意 cute_gemm GEMM
```

机制:kernel 侧的 AG 能力恰是两处共享代码——

1. 调度器 shard-major 旋转解码:`TileScheduler.Params.ag`(tile_scheduler.py:270)非空时,raster/group/serpentine 在单 shard 子问题上进行,`AgParams`(tile_scheduler.py:205)在 `Params.create` 组装(tile_scheduler.py:283-301);`first_shard` 取本 rank,本地 shard 先发(tile_scheduler.py:186-189 docstring)。
2. AB-load warp 的 flag 门:`ag_wait_m_tile`(tile_scheduler.py:124)在 tile 首个 TMA 前自旋等 `flags[shard*chunks+chunk] >= epoch`(模比较,无 wrap 重同步,tile_scheduler.py:172-175);调用点 gemm_sm90.py:1081 与 gemm_sm100.py:1330,1 项已满足门缓存在 gemm_sm90.py:1064。

host 侧 `ag_args`(gemm.py:360 `AllGatherArguments`:flags/epoch/num_shards/first_shard/num_chunks)穿过四个面:

| 面 | 位置 |
|---|---|
| 低层 dispatch `cute_gemm.gemm.gemm` | gemm.py:442;进 plan cache key(:497),`has_ag` 驱动断言组(:724-726)与 SM 断言(:759) |
| epilogue fn frontend(激活/epilogue 变体) | epilogue/frontend.py:698,launch 侧 gemm_runtime/host.py:517→:562 |
| FFI 编译路径 | gemm_tvm_ffi_utils.py:258-276 转 `AgSchedulerArguments`(tile_scheduler.py:181,kernel 侧孪生) |
| 每次 launch 的几何校验 | `validate_ag_geometry`(gemm_tvm_ffi_utils.py:452):M 整除 num_shards,shard 行数整除 tile_M×cluster_M×num_chunks(:460-466);plan 构建前(gemm.py:454-458)与 launch 前(gemm_tvm_ffi_utils.py:483-485)各查一次 |

变体 API 层 gemm_interface.py:892 的 `gemm()` 不透传 ag_args——AG 调用走低层 `gemm` 或 fn frontend。epilogue mod 变体经后者接入的实例在 tests/test_distributed_gemm.py:119-132(`linear_act_mod` + 一个 `ag_args` kwarg);blockscaled 实例在 tests/test_blockscaled_distributed_gemm.py:87-102。world_size==1 走退化路径:无 flags、ag_args=None(all_gather_gemm.py:709-718)。epoch 驻留设备内存(`_bump_epoch`:595),flag 写是 4 字节 CE 拷贝(`_write_flag`:568),host 值不进图——这是整次调用可捕获的前提(all_gather_gemm.py:217-222)。

## BlockScaledAllGatherRunner 的差异

`BlockScaledAllGatherRunner`(all_gather_gemm.py:818)继承父类,只覆 `__init__`(:906)与 `gather`(:975),`gather(X)` 类型对称:torch.Tensor 原样走父类(all_gather_gemm.py:988-991);BlockScaledOperand 加一条 SFA 车道,产出 gathered BlockScaledOperand(all_gather_gemm.py:1012-1020)。SFA 与 qdata 共享同一套 arrival flags:AG 门按 shard 不按张量,门在 tile 全部 TMA(含 SFA)之前(all_gather_gemm.py:844-849)。

| 差异点 | 位置 | 内容 |
|---|---|---|
| 父类缓冲改字节容器 | all_gather_gemm.py:921-924 | super().__init__ 收 (shard_m, row_storage) 的 uint8(fp8 每 K 字节一行,fp4x2 K/2,packed fp6 3K/4) |
| arrival_chunks 恒 1 | all_gather_gemm.py:927 | chunk 边界还须落在 128 行 SF 块上,不值得(:902-903) |
| SFA 缓冲与事件 | all_gather_gemm.py:944-950 | 独立对称 sfa_bufs 同 2-buffer parity,无新增同步(:891-900) |
| shard 级断言 | all_gather_gemm.py:936-937、:957-972 | shard_m % 128、k % (sf_vec_size×4);`_sfa_shard_src` 钉住 (rm, rk, 32, 4, 4) 形状与连续性 |
| gather 顺序 | all_gather_gemm.py:1029-1047 | SFA 发送在 body 期间 enqueue 到父类 push_stream,先于父类 __exit__ 的 [A, flag] 对——同流 CE 序使一个 flag 同时担保 SFA 与 A(:866-877) |

SFA 车道近乎免费的两个事实:blocked SF 布局 128 行 M 块在最外维,每 rank 的 packed shard 是全 tensor 的连续字节切片(all_gather_gemm.py:858-865);同流排序白拿 flag-implies-data。格式描述符与容器契约见 ./blockscaled_api.md。

## 调度器与 dispatch 触点汇总

| 触点 | 位置 | 作用 |
|---|---|---|
| `AllGatherArguments` | gemm.py:360 | host 侧 typed 参数,runner 经 `_ag_args`:523 构造(rank 即 first_shard) |
| plan cache key | gemm.py:497 | 记 `ag_args is not None`,AG 与非 AG 各成 plan |
| 变体合法性 | gemm.py:724-726 | AG 要求 dense、persistent 调度器、split_k==1 |
| SM90+ 断言 | gemm.py:759 | 见下节 |
| 调度参数注入 | gemm_tvm_ffi_utils.py:471(plan_scheduler_args) | gemm.py:651 与 gemm_runtime/host.py:562 两路 launch 都汇到这里 |
| kernel 门 | gemm_sm90.py:1081、gemm_sm100.py:1330 | AB-load warp 自旋,scheduler/MMA/epilogue warp 不受影响(gemm_sm90.py:1072-1075 注释) |

## SM 版本限制

- AG 要求 SM90+:`assert device_capacity[0] >= 9`(gemm.py:759)。SM8x 是唯一没有专门 load warp 的架构(cp.async kernel 无门可挂,gemm.py:757-758 注释);模块自述覆盖 SM90/SM100/SM110(all_gather_gemm.py:2)。
- 断言只在低层 dispatch 的 plan 构建(gemm.py:759)一处。epilogue fn frontend 的编译路径(gemm_runtime/host.py:232 经 `compile_gemm_kernel`)没有等价断言,其 AG 检查仅几何校验(gemm_runtime/host.py:562 → validate_ag_geometry);FFI 工具层自身(make_scheduler_args:gemm_tvm_ffi_utils.py:253)只做参数转换。【推断】fn frontend 路径传 ag_args 时 SM90+ 无显式断言,约束落在 kernel 侧实现上。
- 与 architecture.md「目标硬件差异」一致:SM80 的 split-K、2D unbatched、`b_kn`、AllGather+GEMM 断言均要求 SM90+。

## 参考

- ../architecture.md:「独立 kernel 家族」行与「目标硬件差异」节
- ./blockscaled_api.md:BlockScaledOperand 容器与格式描述符
- 模块 docstring(all_gather_gemm.py:1-249):transport 物理、CUDA-graph 三约束、开销模型的唯一权威表述

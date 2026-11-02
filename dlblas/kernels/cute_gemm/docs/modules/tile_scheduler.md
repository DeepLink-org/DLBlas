# tile 调度器(tile_scheduler.py)

设备侧 tile 调度独立在 `tile_scheduler.py`(1541 行):基类 `TileScheduler`(:250,dense GEMM)加两个子类 `TriangularTileScheduler`(:1067,symmetric 三角)与 `VarlenMTileScheduler`(:1257,varlen-M),统一回答 persistent kernel 的一个问题——每个 CTA 下一步算哪个 tile。对应 [architecture.md](../architecture.md)「SM 设备实现层」(architecture.md:250)与「关键抽象速查」的 `TileScheduler` 行(:294)。

约定:`路径:行号` 相对仓库根(模块内文件省略 cute_gemm/ 前缀)。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 四种持久化模式

`PersistenceMode`(:31-39)枚举 NONE/STATIC/DYNAMIC/CLC。差异集中在两点:work idx 从哪来(硬件 blockIdx 还是软件计数器),取下一个 idx 要不要同步原语。

| 模式 | work idx 来源 | 网格形状 | 取号机制 | 额外资源 |
|---|---|---|---|---|
| NONE(:32) | blockIdx(:54 的 constexpr 除法) | 全问题网格(:468-475) | 无,每 CTA 恰一个 tile | 无 |
| STATIC(:33) | 线性计数器 | 持久网格(:477-491) | `idx += num_persistent_clusters`,纯寄存器递增(:895-896) | 调度 smem + PipelineAsync |
| DYNAMIC(:34) | 线性计数器 | 同 STATIC | 调度 warp 对 semaphore 原子 `INC` 抢号(:903-911);varlen 走 `atomic_add`(:913-915) | 同上 + `(1,)` Int32 semaphore |
| CLC(:39) | 被取消 cluster 的 blockIdx | 与 NONE 同分支(:468) | 硬件 try_cancel,16 B 响应组播进各 CTA smem(:750-767) | 调度 smem(stage 数更大)+ throttle named barrier |

模式选择在 host 装配:`GemmBase.get_scheduler_arguments`(gemm_base.py:1092)的 `:1102-1110`——非 persistent 取 NONE;`arch >= 100 且 use_clc_persistence` 取 CLC;给了 `tile_count_semaphore` 取 DYNAMIC;否则 STATIC。`use_clc_persistence` 默认值:SM100 类 True(gemm_sm100.py:221),SM90 类 False(gemm_sm90.py:166)。semaphore 张量由 host 分配,只供 SM8x/SM90 的 DYNAMIC 消费(gemm.py:964 `device_capacity[0] <= 9`;epilogue 计划同判 gemm_runtime/host.py:464),SM100 动态持久化直接走 CLC。

CLC 的语义:持久 cluster 从 pending pool 硬件抢号(`clusterlaunchcontrol.try_cancel`),响应由硬件组播写进每个 CTA 的 smem 槽——一个槽 16 B = 4×Int32(`SCHED_SLOT_BYTES`,:45),STAS 广播记录与 CLC 响应共用同一布局(:42-44 注释)。work idx 是被取消 cluster 的 x 坐标,batch 是 z 坐标(:408-411),与 NONE 走同一条解码路径;STATIC/DYNAMIC 用 z 坐标当线性计数器(:413-414)。

### host 侧参数

`TileSchedulerOptions`(:226)是 host 传给调度器的选项族:`max_active_clusters`、`raster_order`(默认 Heuristic,:228)、`max_swizzle_size`(:229,即 swizzle 的 group_size)、`tile_count_semaphore`、`batch_idx_permute`、`ag`。静态字段由 `make_scheduler_args`(gemm.py:992)组进 plan 的 `scheduler_static` 槽(gemm.py:243),`tile_count_semaphore` 等 per-call 张量走 launch 参数(gemm.py:509);装配时经 `scheduler_args` 进 `get_scheduler_arguments`。`batch_idx_permute` 是 `(L,)` 的 batch 重排,常规路径作用于 batch 维(:618-623),split-K 下只作用于 l(:627-631)。`TileSchedulerArguments`(:236)是装配产物:新增从操作数推导的 `problem_shape_ntile_mnl`,并冻结 `persistence_mode`(:243)与 `num_split_k`(:246)。

## 共用协议

三个调度器共用一套调用协议,实现在基类,子类不覆写 create/advance/get_current_work(architecture.md:345):

| 阶段 | 方法 | 行为 |
|---|---|---|
| host 参数化 | `to_underlying_arguments` :389 → 各自 `Params.create` :274/:1087/:1278 | 把 `TileSchedulerArguments`(:236)预除成 FastDivmod、解析 raster 启发式(:70)、冻结 persistence_mode |
| host 网格 | `get_grid_shape` :461 | NONE/CLC 全网格;STATIC/DYNAMIC 截到 `max_active_clusters` 一波 |
| 设备构造 | `create` :419 | 从 cluster_idx 解初始 work idx,持调度 smem、PipelineAsync、pipeline_state(:442-457);三个子类共用,靠 `cls` 分派 Params 与 `_cluster_idx_to_work_idx_batch`(:430-432 注释) |
| 首 tile | `initial_work_tile_info` :885 | `_delinearize_work_idx`(:540)展开 |
| 循环 | `advance_to_next_work` :965 + `get_current_work` :670 | 每个 consumer warp 都调;producer 动作由 `is_scheduler_warp` 门控(全 cluster 恰一个 warp 为真) |
| 收尾 | `producer_tail` :1007;CLC 另加 `cancel_pending_tail` :781 | 排空 pipeline;CLC 再排空 padding 尾部 |

work tile 载体是 `WorkTileInfo`(:87):`tile_idx = (pid_m, pid_n, split_idx, batch_idx)`,`num_split_k == 1` 时 split_idx 是静态 None(:91-92 注释)。它自带 MLIR 值协议(:110-120),能跨 persistent 循环边界存活。

有效性判定按模式走三分支(:551-559):NONE 只在 `num_tiles_executed == 0` 时有效(每 CTA 恰一个 tile);CLC 看 `work_idx < M×N 的 cluster 数`(:554-555);其余看 `work_idx < 全问题 cluster 数 × num_split_k`(:556-559)。z 坐标优先取调用方传入的 `bidz`,否则从 special register 现算(:565-569)。

调度器自身也是槽位 pipeline 的用户:producer 视角的状态由 `_producer_state`(:392)从 consumer 状态推导——同 stage、phase 取反(:393-401);`write_work_tile_to_smem` 与 `_issue_clc_query_multicast` 都用它做 `producer_acquire`。

STATIC/DYNAMIC 是两段式生产消费:调度 warp 取号后以 `block_zero_only=True` 解码(:985-987),`write_work_tile_to_smem`(:919)把 4 个 Int32 写进 smem 环形缓冲,多 CTA cluster 经 `store_shared_remote_x4` 组播到各 peer(:954);所有 consumer warp 在 `get_current_work` 里 `consumer_wait` → 读槽 → 异步代理 fence → `consumer_release`(:679-691)。CLC 没有软件写槽:硬件把响应直接落槽,`_get_current_work_clc`(:703)由每个 warp 自己解码并跑 swizzle(:731-732);解码故意放在 release 之后,把可能很贵的 delinearize(varlen 扫描)移出槽位关键路径(:722-725 注释)。

`advance_to_next_work`(:965)按模式分派 producer 动作;consumer 侧一律只推进 `num_tiles_executed` 与 pipeline state(:978-980):

| 模式 | 调度 warp(:983/:991 门控) | 其他 consumer warp |
|---|---|---|
| NONE | 无动作 | 无动作 |
| STATIC/DYNAMIC | `_fetch_next_work_idx`(:889)→ 解码 → `write_work_tile_to_smem`(:984-988) | `get_current_work` 读槽 |
| CLC | throttle 会合后 `_issue_clc_query_multicast`(:1004-1005) | 槽里已是硬件响应,`_get_current_work_clc` 直接解码 |

CLC 查询的硬件路径:`_issue_clc_query_multicast`(:750)中 `lane_idx < cluster_size` 的 lane 对每个 CTA 的 full barrier 做 `mbarrier_arrive_and_expect_tx(16)`(:761-764),`elect_one` 发一条组播 try_cancel(:766-767);响应不经 STAS 转发,由硬件直接落各 CTA 的槽(:754-755 注释)。

kernel 侧消费形态:循环体从 `work_tile.tile_idx` 解出 `(pid_m, pid_n, split_idx, batch_idx)`(gemm_sm90.py:1069-1071),批内偏移交给 VarlenManager。pingpong 的 MMA warp 一次 advance 两个 tile(`advance_count=self.mma_warp_groups`,gemm_sm90.py:1464),pipeline state 由 `advance_iters` 对应快进(:979-980)。

kernel 侧调用点:SM90 AB-load warp 在 :1062-1063 构造并取首 tile,循环尾 :1171-1172 advance + 取下一个,收尾 `producer_tail` :1182。SM100 的调度 warp 构造带 `is_scheduler_warp`(:1527),其余 warp 角色各持只读实例,收尾在 :1539-1546(`cancel_pending_tail` :1546)。

## 三个调度器类与覆写点

五个覆写方法的行号见 architecture.md「多态方法速查」(:337-344)。各类的独有部分:

| 类 | Params.create | 独有方法 / 字段 | 说明 |
|---|---|---|---|
| `TileScheduler` :250 | :274 | `_cluster_id_to_cta_id` :525、`get_split_k_tile_range` :637、`get_combined_batch_idx` :660、`_get_current_work_clc` :703、`_issue_clc_query_multicast` :750、`throttle_producer_commit` :770、`cancel_pending_tail` :781、`_fetch_next_work_idx` :889、`write_work_tile_to_smem` :919 | dense 全功能:split-K、`batch_idx_permute`、AllGather |
| `TriangularTileScheduler` :1067 | :1087 | 模块级 `triangular_idx_to_coord` :1057 | 拒绝 split-K(:1091);cluster 数 = m(m+1)/2(:1102);要求每 cluster 方形 tile(:1068) |
| `VarlenMTileScheduler` :1257 | :1278 | `_get_num_m_blocks` :1421;`grid_may_exceed_work = True` :1258 | 拒绝 split-K(:1281);问题形状的 M 维为 None(:1285) |

`_swizzle_cta` 是 L2 局部性核心:基类版(:494)按 `group_size` 把 fast 维分组成 M×N 小块,奇数组 serpentine 反射(:505-517);三角版在三角索引上分组,sqrt 近似反解行号(:1166-1174);varlen 版多一个运行时参数 `num_clusters_m`(:1380-1381)——M 方向的 cluster 数逐 batch 变化,分组除法只能在运行时做(:1385-1406)。

`grid_may_exceed_work`(:254)是类属性:只有 varlen 调度器会过量供给网格(:1258),其余调度器的退役排水是死代码(:251-253 注释)。

## K 维分解:get_split_k_tile_range

split-K 的 K-tile 切分只属于调度器:`get_split_k_tile_range`(:637)返回 split_idx 拥有的 `[start, start+cnt)`,前 `k_tile_total % num_split_k` 个 split 各多拿一个 k-tile(:653-656),与 CUTLASS 3.x 平衡切分一致(:647)。docstring 写明约束:每个 warp(load/MMA/epilogue/pingpong)都必须从这一个方法推导 k-tile 数,否则 AB 流水线生产/消费计数失步而死锁(:642-645)。超出 `k_tile_total` 的 split 拿空区间但仍跑 epilogue(零贡献),让串行模式的 semaphore 旋转门继续走(:646-648)。

配套两个方法:`get_combined_batch_idx`(:660)是 (l, split) → 组合索引的逆变换,分阶段 split-K 用它当 partials workspace 的 batch 坐标(:663-664 注释)。线性 work idx 里 split 是最快变化分量(:578-584),同一输出 tile 的各 split 时间相邻,省 L2 与 semaphore 等待(:576-577 注释);batch 维 permute 只作用于 l(:627-631)。

## CLC 专属:throttle 与退役排水

- 查询节流:`advance_to_next_work` 的 CLC 分支(:989-1005)发查询前在 `throttle_barrier` 上会合(:1004)。多级 lookahead 会让 cluster 以 CLC 往返节奏(~1us)而非 tile 节奏发查询,过量抢号会饿死其他持久 worker(:994-1003 注释)。单 named barrier 足够:依赖链保证 arrive 严格交替,至多一个 credit 在途。barrier 由 SM100 的首 AB-load warp 每 tile 到达一次(gemm_sm100.py:1310-1316),定义在 gemm_base.py:78-80,构造在 gemm_sm100.py:322-326。
- 退役排水:`cancel_pending_tail`(:781)只在 `grid_may_exceed_work` 时生效(:843-845)。三条不变量(:788-807):只在"解码出的 phantom"(有效 grant、work idx 落在 padding)上排水——grant 单调,一个 phantom 证明真实 work 已全部出池;invalid 响应可能只是竞争下 try_cancel 的虚假失败,据此排水会取消真实 cluster(2026-07 的静默错误)。每次 cancel 串行发、串行等(:800-804)。响应落专用 mailbox,不碰在用环形槽(:805-807);SM100 的调度 smem 为此多留 6 个 Int32(gemm_sm100.py:1016-1017)。上限 `CLC_DRAIN_MAX_CANCELS = 256`(:51)。无排水时首个 varlen 形状 123.75 → 239.9 us,排水本身是必要的(:809-821 实测表)。
- stage 数:SM90 `sched_stage = pingpong ? 2 : 1`(gemm_sm90.py:485);SM100 `= use_clc_persistence ? 3 : 1`(gemm_sm100.py:579)。

## AllGather 触点

`TileSchedulerArguments.ag`(:247)携带 `AgSchedulerArguments`(:181):arrival flags、设备驻留 epoch、分片几何。`Params.create` 把它预除成 `AgParams`(:205),此后 raster/serpentine 全部作用在单 shard 子问题上(:284-304)。解码时线性 id 先按 shard 分、环旋转让本地 shard 先发(:604-611),swizzle 限制在该 shard 内(:612-614)。AB-load warp 在发该 tile 首条 TMA 前调 `ag_wait_m_tile`(:124)等 `flags[shard*chunks+chunk] >= epoch`,带一-entry 满足门缓存跳过多数 tile 的 flag 读(:135-140 注释);调用点 gemm_sm100.py:1328-1334。varlen 与三角调度器的 Params 没有 ag 字段,gemm_sm100.py:1325-1327 注释记录了这一约束。

## varlen 调度与参数族

varlen 下调度器与 `VarlenManager`(varlen_utils.py:28)是同一参数族的并列消费者,调度器不经过 VarlenManager:两者都从 `gemm` 签名的 `cu_seqlens_m` 等参数(gemm_interface.py:922-925)出发,`GemmBase.get_scheduler_arguments` 在 varlen 分支直接把 `varlen_args.mCuSeqlensM` 抄进 `VarlenMTileSchedulerArguments`(gemm_base.py:1164-1182),`Params.create` 存为 `cu_seqlens_m`(:1264/:1315)。VarlenManager 管 tile 内的 A/B/SF 偏移(offset_batch_* 族);调度器只消费 `cu_seqlens_m` 做 work 解码,`A_idx`(gather)与 `cu_seqlens_k` 都不进调度器。

解码算法(`_delinearize_work_idx` :1438):

1. `block_size = tile_m × cluster_m`(:1452),一个 work idx 占据某 batch 的一个 (m-cluster, n) 槽位。
2. 缓存命中(work idx 落在上次解析的 batch 窗口内,:1479-1481)直接复用 batch 边界;未命中进入 warp 协作扫描(:1484-1499):`_get_num_m_blocks`(:1421)每 lane 算一个 batch 的 M-cluster 数(`shuffle_sync_down` 取相邻 cu_seqlens 差,:1429-1430),`warp_prefix_sum` 累计,一次推进 31 个 batch。
3. `popc(ballot)`(:1504-1508)在这批 batch 里定位 work idx 归属,回写缓存字段(:1537-1540)。
4. batch 内偏移交给 `_swizzle_cta(cluster_id_in_problem, num_clusters_m)`(:1532)。

网格是松上界:`total_clusters_m_max = (total_m + L·(block-1)) // block`(:1363),注释给出论证:只凭 (total_m, L) 这是紧上界,再小就有 tile 拿不到 work idx,CLC 下等于算错结果(:1359-1362)。padding 槽由 CLC 的 `cancel_pending_tail` 廉价消化(:1362)。批量窗口缓存存在的量级:CLC 的 steal 在响应 mbarrier 消费前无法预取,无缓存时每 warp 在取号点吃整条扫描链,~3-9% e2e 损失;静态调度下 work idx 是提前一个 tile 已知的寄存器递推,链可与循环体重叠(:1466-1477 注释)。

## 装配:谁构造调度器

| 步骤 | 位置 | 内容 |
|---|---|---|
| 选类 | `GemmBase.get_scheduler_class`(gemm_base.py:1085) | 默认 dense/varlen 二选一;`GemmSymmetricMixin` 覆写返回 `TriangularTileScheduler`(gemm_symmetric.py:73) |
| 选模式 + 收参 | `get_scheduler_arguments`(gemm_base.py:1092,模式 :1102-1110) | 从张量元数据组 `TileSchedulerArguments` / `VarlenMTileSchedulerArguments` |
| 参数与网格 | gemm_sm90.py:777-784(SM100 :981-986) | `to_underlying_arguments` + `get_grid_shape(tile_sched_params, max_active_clusters)` |
| 设备侧工厂 | gemm_sm90.py:1029-1031(SM100 :1268-1274) | `partial(Cls.create, params, sched_data, sched_pipeline)`,SM100 另传 `throttle_barrier`;kernel 内 `TileSchedulerCls()` 即实例(:1062/:1297) |
| 调度 smem | gemm_sm90.py:968-987 | persistent 时建 `make_sched_pipeline`(:1939),分配 `(4, sched_stage)` Int32,RESERVED 分区、16 B 对齐(:977-986 注释) |

`max_active_clusters` 由 `get_max_active_clusters`(cute_dsl_utils.py:128)按 cluster size 查询 HardwareInfo。三角调度要求 `cluster_shape_k == 1` 且方形 cluster tile(:1090-1091)。

## 参考

- [architecture.md](../architecture.md)「SM 设备实现层」「关键抽象速查」「多态方法速查」
- [caching_and_parallel_compilation.md](./caching_and_parallel_compilation.md)——文中 CLC/varlen 实测数字(:809-829)依赖其基准与缓存环境
- [fused_grouped_gemm.md](./fused_grouped_gemm.md)——varlen 调度的主要消费场景
- [varlen_blockscaled_sf_layout.md](./varlen_blockscaled_sf_layout.md)——varlen SF 缓冲的 tile 对齐约定(调度 batch 偏移与 SF 偏移的分工)

import triton
import triton.language as tl


@triton.jit
def grouped_launch_swizzling(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    if (num_pid_m >= GROUP_M) and (num_pid_n >= GROUP_M):
        width = GROUP_M * num_pid_n
        group_id = pid // width
        group_size = tl.minimum(num_pid_m - group_id * GROUP_M, GROUP_M)
        remian_pid = pid - group_id * width
        pid_m = group_id * GROUP_M + (remian_pid % group_size)
        pid_n = (pid % width) // group_size
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    return pid_m, pid_n


'''
    水平分核方式每个任务块编号如下
    [0,  1,  2,  3,  4,  5,  6,  7]
    [8,  9,  10, 11, 12, 13, 14, 15]
    [16, 17, 18, 19, 20, 21, 22, 23]
    [24, 25, 26, 27, 28, 29, 30, 31]
    [32, 33, 34, 35, 36, 37, 38, 39]
    [40, 41, 42, 43, 44, 45, 46, 47]
    [48, 49, 50, 51, 52, 53, 54, 55]
    [56, 57, 58, 59, 60, 61, 62, 63]
    0核处理 0 20 40 60 4块任务
    1核处理 1 21 41 61 4块任务
    2核处理 2 22 42 62 4块任务
    ...
    19核处理 19 39 59 3块任务
    
    大shape下如果使用传统水平分核方式,会有如下问题
    1:同一时间大量核心需要访问同一块左矩阵内存,产生Bank冲突,导致硬件访问效率降低
    2:当完成一整行mat_c运算时,已经将所有右矩阵数据全部使用上,右矩阵较大时会超过L2Cache的容量上限,
      从而导致L2Cache的搬入及换出,此后每行运算都会或多或少产生CacheMiss,导致L2Cche命中率较低,影响
      算子执行效率
    此处使用8 * 8对角线分核方式可以按8 * 8的方块沿对角线方向分核计算,可以很大程度优化上面两点。

    此处以8*8对角线分核为例,实际以BLOCK_TRESHHOLD为tune参数选择最优的阈值
    8 * 8 对角线分核方式中,每8 * 8分格内任务块编号如下
    [0,  8,  16, 24, 32, 40, 48, 56]
    [57, 1,  9,  17, 25, 33, 41, 49]
    [50, 58, 2,  10, 18, 26, 34, 42]
    [43, 51, 59, 3,  11, 19, 27, 35]
    [36, 44, 52, 60, 4,  12, 20, 28]
    [29, 37, 45, 53, 61, 5,  13, 21]
    [22, 30, 38, 46, 54, 62, 6,  14]
    [15, 23, 31, 39, 47, 55, 63, 7]
    
    M轴方向超过8个基本块时,使用对角线分核可以明显减小Bank冲突 
    当右矩阵大小超过L2Cache大小时,采取对角线分核可以提升L2Cache利用率
    所以当矩阵在M和N方向均超过8块时使能对角线分核即可有优化,当右矩阵大小超过L2Cache大小时优化效果尤为明显
'''
@triton.jit
def grouped_launch_diagonal(pid, num_pid_m, num_pid_n, BLOCK_TRESHHOLD: tl.constexpr):
    if (num_pid_m >= BLOCK_TRESHHOLD) and (num_pid_n >= BLOCK_TRESHHOLD):
        # 对角线分核代码实现 
        curThresholdM = BLOCK_TRESHHOLD if pid < (num_pid_m // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * num_pid_n else num_pid_m % BLOCK_TRESHHOLD
        curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
        curThresholdN = BLOCK_TRESHHOLD if pid % (num_pid_n * BLOCK_TRESHHOLD) < (curThresholdM * num_pid_n) // curThresholdM_thresholdN * curThresholdM_thresholdN else num_pid_n % BLOCK_TRESHHOLD
        localRelativeBlock = pid % (BLOCK_TRESHHOLD * num_pid_n) % (BLOCK_TRESHHOLD * curThresholdM)
        task_m_idx = localRelativeBlock % curThresholdM + pid // (BLOCK_TRESHHOLD * num_pid_n) * BLOCK_TRESHHOLD
        # 求最小公倍数，方便求基本块的坐标
        x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
        while y != 0:
            x, y = y, x % y
        lcm = curThresholdM * curThresholdN // x
        task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + pid % (BLOCK_TRESHHOLD * num_pid_n) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
    else:
        task_m_idx = pid // num_pid_n
        task_n_idx = pid % num_pid_n
    return task_m_idx, task_n_idx
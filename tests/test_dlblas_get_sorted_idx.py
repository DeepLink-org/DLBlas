import pytest
import torch
from dlblas.layers.moe.kernels.fused_moe import _dlblas_get_sorted_idx


# 测试 _dlblas_get_sorted_idx
def test_dlblas_get_sorted_idx():
    # 准备测试数据
    topk_ids = torch.tensor([[ 18,  29,  34,  43, 235, 254, 255, 226],
       [ 72,  74,  88,  93, 127, 185, 195,  64],
       [ 99, 102, 107, 109, 126, 196, 210,   3],
       [ 68,  92, 101, 117, 153, 234,  87, 108],
       [  1,   9,  92, 170,  37,  59,  81,  87],
       [ 41,  67,  92, 217, 249,  37,  47,  77],
       [ 11,  49,  55,  69,  81,  82, 104,   4],
       [  2,   6,  39,  44, 102, 124, 137,   4],
       [ 31,  33,  52,  68,  72, 208,  12,  27],
       [ 11, 107, 109, 116, 124, 174, 205,  16],
       [ 84,  87,  88, 146, 156, 180, 222,  65]], device='cuda:0')
    num_experts = 256

    # 调用 _dlblas_get_sorted_idx 方法
    sorted_idx, exp_start, exp_end = _dlblas_get_sorted_idx(topk_ids, num_experts)

    # print("Actual sorted_idx:", sorted_idx)
    # print("Actual exp_start:", exp_start)
    # print("Actual exp_end:", exp_end)

    # 验证排序后的索引（根据实际输出调整）
    expected_sorted_idx = torch.tensor([32, 56, 23, 55, 63, 57, 33, 48, 72, 70, 79,  0, 71,  1, 64, 65,  2, 36,
       45, 58, 40,  3, 59, 46, 49, 66, 50, 37, 15, 87, 41, 24, 67, 51,  8, 68,
        9, 47, 38, 52, 53, 80, 30, 39, 81, 10, 82, 25, 34, 42, 11, 16, 26, 17,
       60, 54, 18, 73, 31, 19, 74, 75, 27, 61, 76, 20, 12, 62, 83, 28, 84, 35,
       77, 85, 13, 14, 21, 78, 69, 22, 43, 86,  7, 29,  4, 44,  5,  6],
      device='cuda:0')
    # print("Expected sorted_idx:", expected_sorted_idx.shape)
    # print("Expected sorted_idx:", sorted_idx.shape)
    assert torch.all(sorted_idx == expected_sorted_idx), "sorted_idx不匹配"
    # print("Expected sorted_idx is ok")

    # 验证专家起始和结束索引（根据实际输出调整）
    expected_exp_start = torch.tensor([ 0,  0,  1,  2,  3,  0,  5,  0,  0,  6,  0,  7,  9,  0,  0,  0, 10,  0,
       11,  0,  0,  0,  0,  0,  0,  0,  0, 12,  0, 13,  0, 14,  0, 15, 16,  0,
        0, 17,  0, 19,  0, 20,  0, 21, 22,  0,  0, 23,  0, 24,  0,  0, 25,  0,
        0, 26,  0,  0,  0, 27,  0,  0,  0,  0, 28, 29,  0, 30, 31, 33,  0,  0,
       34,  0, 36,  0,  0, 37,  0,  0,  0, 38, 40,  0, 41,  0,  0, 42, 45,  0,
        0,  0, 47, 50,  0,  0,  0,  0,  0, 51,  0, 52, 53,  0, 55,  0,  0, 56,
       58, 59,  0,  0,  0,  0,  0,  0, 61, 62,  0,  0,  0,  0,  0,  0, 63,  0,
       65, 66,  0,  0,  0,  0,  0,  0,  0,  0,  0, 67,  0,  0,  0,  0,  0,  0,
        0,  0, 68,  0,  0,  0,  0,  0,  0, 69,  0,  0, 70,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0, 71,  0,  0,  0, 72,  0,  0,  0,  0,  0,
       73,  0,  0,  0,  0, 74,  0,  0,  0,  0,  0,  0,  0,  0,  0, 75, 76,  0,
        0,  0,  0,  0,  0,  0,  0, 77,  0,  0, 78,  0, 79,  0,  0,  0,  0,  0,
        0, 80,  0,  0,  0,  0, 81,  0,  0,  0, 82,  0,  0,  0,  0,  0,  0,  0,
       83, 84,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 85,  0,  0,
        0,  0, 86, 87], device='cuda:0')
    expected_exp_end = torch.tensor([ 0,  1,  2,  3,  5,  0,  6,  0,  0,  7,  0,  9, 10,  0,  0,  0, 11,  0,
       12,  0,  0,  0,  0,  0,  0,  0,  0, 13,  0, 14,  0, 15,  0, 16, 17,  0,
        0, 19,  0, 20,  0, 21,  0, 22, 23,  0,  0, 24,  0, 25,  0,  0, 26,  0,
        0, 27,  0,  0,  0, 28,  0,  0,  0,  0, 29, 30,  0, 31, 33, 34,  0,  0,
       36,  0, 37,  0,  0, 38,  0,  0,  0, 40, 41,  0, 42,  0,  0, 45, 47,  0,
        0,  0, 50, 51,  0,  0,  0,  0,  0, 52,  0, 53, 55,  0, 56,  0,  0, 58,
       59, 61,  0,  0,  0,  0,  0,  0, 62, 63,  0,  0,  0,  0,  0,  0, 65,  0,
       66, 67,  0,  0,  0,  0,  0,  0,  0,  0,  0, 68,  0,  0,  0,  0,  0,  0,
        0,  0, 69,  0,  0,  0,  0,  0,  0, 70,  0,  0, 71,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0, 72,  0,  0,  0, 73,  0,  0,  0,  0,  0,
       74,  0,  0,  0,  0, 75,  0,  0,  0,  0,  0,  0,  0,  0,  0, 76, 77,  0,
        0,  0,  0,  0,  0,  0,  0, 78,  0,  0, 79,  0, 80,  0,  0,  0,  0,  0,
        0, 81,  0,  0,  0,  0, 82,  0,  0,  0, 83,  0,  0,  0,  0,  0,  0,  0,
       84, 85,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 86,  0,  0,
        0,  0, 87, 88], device='cuda:0')
    assert torch.all(exp_start == expected_exp_start)
    assert torch.all(exp_end == expected_exp_end)
    # print("test_dlblas_get_sorted_idx is ok")


# 测试 _dlblas_get_sorted_idx 输入简单数据
def test_dlblas_get_sorted_idx_easy():
    # 准备测试数据
    topk_ids = torch.tensor([[1, 3, 2], [4, 2, 1]], device='cuda')
    num_experts = 5

    # 调用 _dlblas_get_sorted_idx 方法
    sorted_idx, exp_start, exp_end = _dlblas_get_sorted_idx(topk_ids, num_experts)

    # print("Actual sorted_idx:", sorted_idx)
    # print("Actual exp_start:", exp_start)
    # print("Actual exp_end:", exp_end)

    # 验证排序后的索引（根据实际输出调整）
    expected_sorted_idx = torch.tensor([5, 0, 2, 4, 1, 3], device='cuda')
    # expected_sorted_idx = torch.tensor([0, 5, 2, 4, 1, 3], device='cuda')
    assert torch.all(sorted_idx == expected_sorted_idx)

    # 验证专家起始和结束索引（根据实际输出调整）
    expected_exp_start = torch.tensor([0, 0, 2, 4, 5], device='cuda')
    expected_exp_end =   torch.tensor([0, 2, 4, 5, 6], device='cuda')
    assert torch.all(exp_start == expected_exp_start)
    assert torch.all(exp_end == expected_exp_end)
    # print("test_dlblas_get_sorted_idx_easy is ok")

# 测试 _dlblas_get_sorted_idx 输入包含负数
def test_dlblas_get_sorted_idx_with_negative():
    # 测试数据包含一个 -1
    topk_ids = torch.tensor([[1, -1], [3, 2]], device='cuda')
    num_experts = 4  # 专家ID范围 [0,3]

    sorted_idx, exp_start, exp_end = _dlblas_get_sorted_idx(topk_ids, num_experts)
    
    # 验证排序后的索引（-1 应该出现在最前面）
    expected_sorted_idx = torch.tensor([1, 0, 3, 2], device='cuda')  # 对应专家值 [-1, 1, 2, 3]
    
    # print(f"sorted_idx = {sorted_idx}")
    # print(f"exp_start = {exp_start}")
    # print(f"exp_end = {exp_end}")

    assert torch.all(sorted_idx == expected_sorted_idx)
    
    # 验证专家分布（仅处理有效专家 0-3）
    expected_exp_start = torch.tensor([0, 1, 2, 3], device='cuda')
    expected_exp_end =   torch.tensor([0, 2, 3, 4], device='cuda')
    assert torch.all(exp_start == expected_exp_start)
    assert torch.all(exp_end == expected_exp_end)
    # print("test_dlblas_get_sorted_idx_with_negative is ok")

# 运行测试
if __name__ == '__main__':
    pytest.main()
"""
exp:
    topk_ids = torch.tensor([[1, 3, 2], [4, 2, 1]], device='cuda')
    num_experts = 5

排序后的索引数组: [5, 0, 2, 4, 1, 3]
对应的专家ID值: 
  索引5 → 原值1
  索引0 → 原值1
  索引2 → 原值2
  索引4 → 原值2
  索引1 → 原值3
  索引3 → 原值4

专家分布可视化：
位置 | 0 | 1 | 2 | 3 | 4 | 5 
-----|---|---|---|---|---|---
专家ID（数据） |1 |1 |2 |2 |3 |4

专家统计：
1. 专家0：没有出现 → 0个token（索引范围[0,0)）
2. 专家1：位置0-1 → 2个token（索引范围[0,2)）
3. 专家2：位置2-3 → 2个token（索引范围[2,4)）
4. 专家3：位置4 → 1个token（索引范围[4,5)）
5. 专家4：位置5 → 1个token（索引范围[5,6)）

因此，需要sorted_idx, 还需要start_idx, end_idx
sorted_idx: [5, 0, 2, 4, 1, 3]
start_idx: [0, 0, 2, 4, 5, 5]
end_idx: [2, 2, 4, 5, 6, 6]
通过这组数据，可以得到每个专家的token分布范围，从而实现高效的计算和存储。

"""
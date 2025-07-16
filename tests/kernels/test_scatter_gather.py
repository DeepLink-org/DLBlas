# Copyright (c) 2025, DeepLink.
import random
import torch

from dlblas.kernels.fused_moe_v3 import ep_scatter, ep_gather


class TestScatterGather:

    def test_scatter(self):

        # scatter
        block_size = 128
        num_recv_tokens_per_expert_list = [0] * 32
        num_recv_tokens_per_expert_list[6] = 128
        num_recv_tokens_per_expert_list[7] = 128
        num_recv_tokens_per_expert_list[8] = 128
        num_recv_tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, dtype=torch.int, device="cuda")

        all_tokens = sum(num_recv_tokens_per_expert_list)
        m_indices_ref = torch.empty(all_tokens, device="cuda", dtype=torch.int32)
        m_indices = torch.empty(all_tokens, device="cuda", dtype=torch.int32)

        recv_x = torch.randn((7, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        recv_x_scale = torch.randn((7, 4096 // block_size), device="cuda", dtype=torch.float32)

        recv_topk_id = torch.ones((7, 8), device="cuda", dtype=torch.int32) * -1
        recv_topk_weights = torch.zeros((7, 8), device="cuda", dtype=torch.float)
        for i in range(7):
            for j in range(4):
                idx = random.randint(0, 7)
                expert_id = random.randint(6, 8)
                recv_topk_id[i][idx] = expert_id
                recv_topk_weights[i][idx] = random.randint(0, 10) / 10.0

        output_indexs = torch.zeros_like(recv_topk_id)
        output_tensor = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        output_tensor_ref = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)

        output_tensor_scale = torch.zeros((all_tokens, 4096 // block_size), device="cuda", dtype=torch.float32)
        output_tensor_scale_ref = torch.zeros((all_tokens, 4096 // block_size), device="cuda", dtype=torch.float32)

        expert_start_loc = torch.cumsum(torch.tensor([0] + num_recv_tokens_per_expert_list[:-1], device="cuda"), dim=0)

        cur = 0
        for i, k in enumerate(num_recv_tokens_per_expert_list):
            m_indices_ref[cur : cur + k] = i
            cur += k

        ep_scatter(
            recv_x,
            recv_x_scale,
            recv_topk_id,
            num_recv_tokens_per_expert,
            expert_start_loc,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_indexs,
        )
        assert torch.allclose(m_indices, m_indices_ref, atol=1e-2, rtol=0)

        for i in range(recv_topk_id.shape[0]):
            for j in range(recv_topk_id.shape[1]):
                if recv_topk_id[i][j] >= 0:
                    dst = output_indexs[i][j]
                    output_tensor_ref[dst][:] = recv_x[i][:]
                    output_tensor_scale_ref[dst][:] = recv_x_scale[i][:]

        assert torch.allclose(output_tensor.to(torch.float), output_tensor_ref.to(torch.float), atol=1e-2, rtol=0)
        assert torch.allclose(output_tensor_scale, output_tensor_scale_ref, atol=1e-2, rtol=0)


    def test_gather(self):

        #### gather
        recv_x = torch.randn((7, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        num_recv_tokens_per_expert_list = [0] * 32
        num_recv_tokens_per_expert_list[6] = 128
        num_recv_tokens_per_expert_list[7] = 128
        num_recv_tokens_per_expert_list[8] = 128
        all_tokens = sum(num_recv_tokens_per_expert_list)

        recv_topk_id = torch.ones((7, 8), device="cuda", dtype=torch.int32) * -1
        recv_topk_weights = torch.zeros((7, 8), device="cuda", dtype=torch.float)
        for i in range(7):
            for j in range(4):
                idx = random.randint(0, 7)
                expert_id = random.randint(6, 8)
                recv_topk_id[i][idx] = expert_id
                recv_topk_weights[i][idx] = random.randint(0, 10) / 10.0
        output_indexs = torch.zeros_like(recv_topk_id)

        gather_out_ref = torch.zeros_like(recv_x, device="cuda", dtype=torch.bfloat16)
        gather_out = torch.empty_like(recv_x, device="cuda", dtype=torch.bfloat16)
        gather_input = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.bfloat16)
        for i in range(recv_topk_id.shape[0]):
            for j in range(recv_topk_id.shape[1]):
                if recv_topk_id[i][j] >= 0:
                    dst = output_indexs[i][j]
                    gather_out_ref[i][:] += gather_input[dst][:] * recv_topk_weights[i][j]
        ep_gather(gather_input, recv_topk_id, recv_topk_weights, output_indexs, gather_out)
        assert torch.allclose(gather_out, gather_out_ref, atol=1e-2, rtol=0)

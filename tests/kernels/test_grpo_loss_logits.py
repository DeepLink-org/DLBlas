
import unittest
import torch
from  dlblas.kernels.grpo_compute_loss_logits import grpo_compute_loss_torch, grpo_loss_triton

class TestGRPOFunctionality(unittest.TestCase):
    
    def setUp(self):
      
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.skipTest("CUDA is not available, skipping Triton tests.")

    def _run_and_compare(self, BL, V, dtype, **kwargs):
        
       
        new_logits = torch.randn((BL, V), device=self.device, dtype=dtype)
        ref_logits = torch.randn((BL, V), device=self.device, dtype=dtype)
        old_logits = torch.randn((BL, V), device=self.device, dtype=dtype)
        input_ids = torch.randint(0, V, (BL,), device=self.device)
        advantages = torch.randn((BL,), device=self.device, dtype=dtype)
        mask = torch.randint(0, 2, (BL,), device=self.device).bool()

       
        loss_pt, comp_pt, kl_pt, loss_i_pt, kl_i_pt = grpo_compute_loss_torch(
            new_logits, ref_logits, old_logits, input_ids, advantages, mask, **kwargs
        )

       
        loss_tr, comp_tr, kl_tr, loss_i_tr, kl_i_tr = grpo_loss_triton(
            new_logits, ref_logits, old_logits, input_ids, advantages, mask, **kwargs
        )
        
    
        atol = 1e-2 if dtype == torch.float16 else 1e-5
        rtol = 1e-2 if dtype == torch.float16 else 1e-5

        self.assertTrue(torch.allclose(loss_pt, loss_tr, atol=atol, rtol=rtol), "Aggregated Loss mismatch")
        self.assertTrue(torch.allclose(kl_pt, kl_tr, atol=atol, rtol=rtol), "Mean KL mismatch")
        self.assertTrue(torch.allclose(comp_pt, comp_tr), "Completion Length mismatch")
        self.assertTrue(torch.allclose(loss_i_pt, loss_i_tr, atol=atol, rtol=rtol), "Per-token Loss mismatch")
        self.assertTrue(torch.allclose(kl_i_pt, kl_i_tr, atol=atol, rtol=rtol), "Per-token KL mismatch")

    def test_base_case_fp32(self):
      
        print("\nRunning: test_base_case_fp32")
        self._run_and_compare(BL=1024, V=2048, dtype=torch.float32, delta=5.0, beta=0.1, temperature=1.0)

    def test_base_case_fp16(self):
        
        print("\nRunning: test_base_case_fp16")
        self._run_and_compare(BL=1024, V=2048, dtype=torch.float16, delta=5.0, beta=0.1, temperature=1.0)
        
    def test_no_delta_clipping(self):
       
        print("\nRunning: test_no_delta_clipping")
        self._run_and_compare(BL=512, V=1024, dtype=torch.float16, delta=None)

    def test_zero_beta(self):
        """测试 beta=0 的情况，即没有 KL 惩罚"""
        print("\nRunning: test_zero_beta")
        self._run_and_compare(BL=512, V=1024, dtype=torch.float16, beta=0.0)

    def test_temperature_scaling(self):
        """测试 temperature != 1.0 的情况"""
        print("\nRunning: test_temperature_scaling")
        self._run_and_compare(BL=512, V=1024, dtype=torch.float16, temperature=2.0)

    def test_all_masked(self):
        """边缘情况测试：所有 token 都被掩码"""
        print("\nRunning: test_all_masked")
        BL, V, dtype = 256, 512, torch.float16
        new_logits = torch.randn((BL, V), device=self.device, dtype=dtype)
        ref_logits = torch.randn((BL, V), device=self.device, dtype=dtype)
        old_logits = torch.randn((BL, V), device=self.device, dtype=dtype)
        input_ids = torch.randint(0, V, (BL,), device=self.device)
        advantages = torch.randn((BL,), device=self.device, dtype=dtype)
        mask = torch.zeros((BL,), device=self.device).bool() # 全 False 掩码

        loss_pt, _, kl_pt, _, _ = grpo_compute_loss_torch(new_logits, ref_logits, old_logits, input_ids, advantages, mask)
        loss_tr, _, kl_tr, _, _ = grpo_loss_triton(new_logits, ref_logits, old_logits, input_ids, advantages, mask)

     
        self.assertTrue(torch.allclose(loss_pt, torch.tensor(0.0, device=self.device)), "Torch loss should be 0 when all masked")
        self.assertTrue(torch.allclose(loss_tr, torch.tensor(0.0, device=self.device)), "Triton loss should be 0 when all masked")
        self.assertTrue(torch.allclose(kl_pt, torch.tensor(0.0, device=self.device)), "Torch KL should be 0 when all masked")
        self.assertTrue(torch.allclose(kl_tr, torch.tensor(0.0, device=self.device)), "Triton KL should be 0 when all masked")

if __name__ == '__main__':
    unittest.main()

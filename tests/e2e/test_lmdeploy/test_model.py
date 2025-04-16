# Copyright (c) 2025, DeepLink.
# For https://github.com/InternLM/lmdeploy/tree/v0.6.1
import torch

from dlblas.kernels.activation import silu_and_mul
from dlblas.kernels.apply_rotary_pos_emb import apply_rotary_pos_emb
from dlblas.kernels.fill_kv_cache import fill_kv_cache
from dlblas.kernels.multinomial_sampling import multinomial_sampling
from dlblas.kernels.paged_attention import paged_attention_fwd
from dlblas.kernels.rms_norm import rms_norm


def _patch_lmdeploy():
    import lmdeploy.pytorch.kernels.cuda as lmdeploy_kernels

    DEFAULT_PATCH_LIST = [
        'fill_kv_cache',
        'apply_rotary_pos_emb',
        'paged_attention_fwd',
        'rms_norm',
        'multinomial_sampling',
        'silu_and_mul',
    ]

    def try_patch(op: str):

        def patch_fill_kv_cache():
            lmdeploy_kernels.fill_kv_cache = fill_kv_cache

        def patch_apply_rotary_pos_emb():
            lmdeploy_kernels.apply_rotary_pos_emb = apply_rotary_pos_emb

        def patch_paged_attention_fwd():
            lmdeploy_kernels.paged_attention_fwd = paged_attention_fwd

        def patch_rms_norm():
            lmdeploy_kernels.rms_norm = rms_norm

        def patch_multinomial_sampling():
            lmdeploy_kernels.multinomial_sampling = multinomial_sampling

        def patch_silu_and_mul():
            import lmdeploy.pytorch.kernels.cuda.activation as activation

            activation.silu_and_mul = silu_and_mul

        try:
            locals()[f"patch_{op}"]()
            print(f"patched dlblas implementation of {op}\n", end='')
        except KeyError:
            print(
                f"unknow op: {op}, supported ops: {DEFAULT_PATCH_LIST}\n",
                end='',
            )
        except AttributeError:
            print(f"op {op} is not implemented in dlblas\n", end='')

    for op in DEFAULT_PATCH_LIST:
        try_patch(op)


_patch_lmdeploy()


def run_pipeline_chat_test(device_type, ):
    tp = 1
    hf_path = '/home/work/models_all/ZhipuAI/glm-4v-9b/'
    # hf_path = "/home/work/models_all/ZhipuAI/glm-4-9b/"
    # hf_path = "/home/work/models_all/ZhipuAI/cogvlm2-llama3-chat-19B/"
    # hf_path = "/home/work/models_all/ZhipuAI/cogvlm-chat/"
    # hf_path = "/home/work/models_all/swift/llava-1___5-7b-hf/"
    # hf_path = "/home/work/models_all/Shanghai_AI_Laboratory/internlm-7b/"
    # hf_path = "/home/work/models_all/Qwen/Qwen1.5-MoE-A2.7B-Chat/"
    # hf_path = "/home/work/models_all/OpenGVLab/InternVL-Chat-V1-5/"
    # hf_path = "/home/work/models_all/OpenBMB/MiniCPM3-4B/"
    # hf_path = "/home/work/models_all/LLM-Research/Phi-3-vision-128k-instruct/"
    # hf_path = "/home/work/models_all/LLM-Research/Phi-3-mini-4k-instruct/"
    # hf_path = "/home/work/models_all/LLM-Research/Phi-3___5-vision-instruct/"
    # hf_path = "/home/work/models_all/LLM-Research/Phi-3___5-MoE-instruct/"
    # hf_path = "/home/work/models_all/LLM-Research/Phi-3___5-mini-instruct/"
    # hf_path = "/home/work/models_all/01ai/Yi-6B/"
    # hf_path = "/workspace/volume/shangda/share/llm_models/Shanghai_AI_Laboratory/internlm2-chat-7b/"  # done
    # hf_path = '/workspace/volume/shangda/share/llm_models/Shanghai_AI_Laboratory/internlm2_5-7b/'
    # hf_path = "/workspace/volume/shangda/share/llm_models/Qwen/Qwen2-7B/"  # done
    # hf_path = '/workspace/volume/shangda/share/llm_models/shakechen/Llama-2-7b-hf/'
    from lmdeploy import PytorchEngineConfig, pipeline
    from lmdeploy.messages import GenerationConfig
    backend_config = PytorchEngineConfig(tp=tp, device_type=device_type, eager_mode=True)
    print('backend_config: ', backend_config)
    pipe = pipeline(hf_path, backend_config=backend_config)
    gen_config = GenerationConfig(do_sample=False, top_k=1)
    response = pipe(
        [
            'Please introduce Shanghai.'
            # "给我一首中文诗，需要添加标点符号，请用中文回答Give me a Chinese poem in Chinese"
        ],
        do_preprocess=False,
        gen_config=gen_config,
    )[0].text
    print(response)
    # assert "诗" in response and "《" in response and "》" in response
    del pipe
    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed = 1024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import pytest

    # pytest.main(['-vs', '/workspace/volume/shangda/zhaochaoxing/work/Triton/python/dlBLAS/tests/kernels/test_activation.py'])
    # pytest.main(['-vs', '/workspace/volume/shangda/zhaochaoxing/work/Triton/python/dlBLAS/tests/kernels/test_apply_rotary.py'])
    # pytest.main(['-vs', '/workspace/volume/shangda/zhaochaoxing/work/Triton/python/dlBLAS/tests/kernels/test_fill_kv_cache.py'])
    # pytest.main(['-vs', '/workspace/volume/shangda/zhaochaoxing/work/Triton/python/dlBLAS/tests/kernels/test_rms_norm.py'])
    # pytest.main(['-vs', '/workspace/volume/shangda/zhaochaoxing/work/Triton/python/dlBLAS/tests/kernels/test_paged_attention.py'])
    # pytest.main(['-vs', '/workspace/volume/shangda/zhaochaoxing/work/Triton/python/dlBLAS/tests/kernels/test_multinomial_sampling.py'])
    # device_type = 'mlu'
    device_type = 'cuda'
    if 'mlu' == device_type:
        import torch_mlu
        import torch_mlu.utils.gpu_migration
    run_pipeline_chat_test('cuda')
    print('sucessfully!')

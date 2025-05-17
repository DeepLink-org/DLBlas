#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <tuple>
#include <vector>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

using fptr_t = int64_t;

std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t n_share_experts_fusion,
    double routed_scaling_factor);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                              int64_t block_size,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor experts_ids,
                              torch::Tensor num_tokens_post_pad);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void per_token_group_quant_fp8(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max);

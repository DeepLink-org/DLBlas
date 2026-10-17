# -*- coding: utf-8 -*-
"""
Task 02: Equivariant Tensor Product - Ascend NPU Implementation

Key migration fix: o3.wigner_3j uses complex128 internally (unsupported on
Ascend NPU), so all Wigner 3j symbols are precomputed on CPU during __init__
and registered as float32 buffers.

Hardware: Huawei Ascend 910B2C
Framework: PyTorch 2.8.0 + torch_npu 2.8.0
"""

import os

os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")

import math
import torch
import torch.nn as nn
from collections import OrderedDict
from math import sqrt
from typing import List

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

import e3nn
from e3nn import o3
from e3nn.o3._tensor_product._instruction import Instruction
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin
from opt_einsum_fx import optimize_einsums_full
from sympy.physics.wigner import wigner_6j
from torch import fx

# ==============================================================================
# Helper Functions
# ==============================================================================


def slices_basis(irreps):
    s = []
    i = 0
    for _, ir in irreps:
        d = 2 * ir.l + 1
        s.append(slice(i, i + d))
        i += d
    return s


def _sum_tensors(xs: List[torch.Tensor], shape, like: torch.Tensor):
    if len(xs) == 0:
        return like.new_zeros(shape)

    out = xs[0]
    for x in xs[1:]:
        out = out + x
    return out


def get_path_norm(irreps_in1, irreps_in2, irreps_out):
    irreps_in1 = e3nn.o3.Irreps(irreps_in1)
    irreps_in2 = e3nn.o3.Irreps(irreps_in2)
    irreps_out = e3nn.o3.Irreps(irreps_out)

    counter = {}
    for _, (_, ir_1) in enumerate(irreps_in1):
        for _, (_, ir_2) in enumerate(irreps_in2):
            for _, (_, ir_out) in enumerate(irreps_out):
                if ir_out in ir_1 * ir_2:
                    counter[ir_out.l] = counter.get(ir_out.l, 0) + 1

    buffer = []
    for _, ir in irreps_out:
        buffer.append(
            torch.ones(2 * ir.l + 1, dtype=torch.float32) * counter.get(ir.l, 0)
        )

    return torch.cat(buffer, dim=0)


def _codegen_graph(si1, si2, so, sins, sim_tp=None, info=None):
    """
    Build FX graph for tensor product forward pass.

    Important for Ascend NPU:
    o3.wigner_3j may internally touch complex128, so all Wigner 3j values
    are computed on CPU during graph construction and stored as float32 buffers.
    """
    graph = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = OrderedDict()

    x1s = fx.Proxy(graph.placeholder("x1", torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder("x2", torch.Tensor), tracer=tracer)
    weights = fx.Proxy(graph.placeholder("w", torch.Tensor), tracer=tracer)

    output_shape = x1s.shape[:-2]

    x1s = x1s.reshape(-1, si1.dim // si1[0].mul, si1[0].mul)
    x2s = x2s.reshape(-1, si2.dim // si2[0].mul, si2[0].mul)
    batch_numel = x1s.shape[0]

    x1_list = [
        x1s[:, i].reshape(batch_numel, m.ir.dim, m.mul)
        for i, m in zip(slices_basis(si1), si1)
    ]
    x2_list = [
        x2s[:, i].reshape(batch_numel, m.ir.dim, m.mul)
        for i, m in zip(slices_basis(si2), si2)
    ]

    outputs = []
    flat_weight_index = 0

    for idx, ins in enumerate(sins):
        mi1 = si1[ins.i_in1]
        mi2 = si2[ins.i_in2]
        mo = so[ins.i_out]

        x1 = x1_list[ins.i_in1]
        x2 = x2_list[ins.i_in2]

        w3j_name = f"_w3j_{mi1.ir.l}_{mi2.ir.l}_{mo.ir.l}"
        w3j = fx.Proxy(graph.get_attr(w3j_name), tracer=tracer)

        if ins.has_weight:
            if sim_tp is not None:
                w = weights[
                    sim_tp.get_weight_byL1L2L3(info[idx][0], info[idx][1], info[idx][2])
                ].reshape(tuple(ins.path_shape))
            else:
                w = weights[
                    flat_weight_index : flat_weight_index + prod(ins.path_shape)
                ].reshape(tuple(ins.path_shape))

            flat_weight_index += prod(ins.path_shape)

        if ins.connection_mode == "uvw":
            xx = torch.einsum("ziu,zjv,ijk->zku", x1, x2, w3j)
            w = w.squeeze()
            result = torch.matmul(xx, w)

        elif ins.connection_mode == "uvu":
            if ins.has_weight:
                xx = torch.einsum("ziu,zjv,ijk->zkuv", x1, x2, w3j)
                result = torch.einsum("uv,zkuv->zku", w, xx)
            else:
                result = torch.einsum("ziu,zjv,ijk->zku", x1, x2, w3j)

        elif ins.connection_mode == "uuu":
            result = torch.einsum("ziu,zju,ijk->zku", x1, x2, w3j)

        else:
            raise RuntimeError(f"Unsupported connection_mode: {ins.connection_mode}")

        result = ins.path_weight * result
        outputs.append(result.reshape(batch_numel, mo.ir.dim, mo.mul))

        if len(w3j.node.users) == 0:
            graph.erase_node(w3j.node)
        elif w3j_name not in constants:
            constants[w3j_name] = (
                o3.wigner_3j(mi1.ir.l, mi2.ir.l, mo.ir.l)
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )

    outputs = [
        _sum_tensors(
            [out for ins, out in zip(sins, outputs) if ins.i_out == i_out],
            shape=(batch_numel, mo.ir.dim, mo.mul),
            like=x1s,
        )
        for i_out, mo in enumerate(so)
        if mo.mul > 0
    ]

    outputs = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
    outputs = outputs.reshape(output_shape + (outputs.shape[-2], outputs.shape[-1]))

    graph.output(outputs.node, torch.Tensor)
    graph.lint()

    constants_root = torch.nn.Module()
    for key, value in constants.items():
        constants_root.register_buffer(key, value)

    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_forward")

    batchdim = 4
    example_inputs = (
        torch.zeros((batchdim, si1.dim // si1[0].mul, si1[0].mul), dtype=torch.float32),
        torch.zeros((batchdim, si2.dim // si2[0].mul, si2[0].mul), dtype=torch.float32),
        torch.zeros(flat_weight_index, dtype=torch.float32),
    )

    graphmod = optimize_einsums_full(graphmod, example_inputs)
    return graphmod


# ==============================================================================
# Model Classes
# ==============================================================================


class Simple_TensorProduct_oTchannel(torch.nn.Module, CodeGenMixin):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple] = None,
        learnable_weight=None,
        connection_mode="uvu",
        reduce_same_order=False,
        in1_var=None,
        in2_var=None,
        out_var=None,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights=True,
        path_weight_sqrt=True,
        rescale=True,
        use_bias=False,
    ):
        super().__init__()

        self.rescale = rescale
        self.use_bias = use_bias

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)

        if instructions is None:
            instructions, irreps_output = self._get_instruction(
                irreps_in1,
                irreps_in2,
                irreps_out,
                learnable_weight=learnable_weight,
                connection_mode=connection_mode,
            )
            self.irreps_out = irreps_output

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]

        self.instructions = []
        for i_in1, i_in2, i_out, cm, hw, pw in instructions:
            path_shape = {
                "uvw": (
                    self.irreps_in1[i_in1].mul,
                    self.irreps_in2[i_in2].mul,
                    self.irreps_out[i_out].mul,
                ),
                "uvu": (
                    self.irreps_in1[i_in1].mul,
                    self.irreps_in2[i_in2].mul,
                ),
                "uuu": (self.irreps_in1[i_in1].mul,),
            }[cm]

            self.instructions.append(
                Instruction(i_in1, i_in2, i_out, cm, hw, pw, path_shape)
            )

        if in1_var is None:
            in1_var = [1.0] * len(self.irreps_in1)
        if in2_var is None:
            in2_var = [1.0] * len(self.irreps_in2)
        if out_var is None:
            out_var = [1.0] * len(self.irreps_out)

        normalization_coefficients = []
        for ins in self.instructions:
            mul_ir_out = self.irreps_out[ins.i_out]

            alpha = 1.0
            if irrep_normalization == "component":
                alpha = mul_ir_out.ir.dim

            x = 1.0
            if path_normalization == "element":
                x = sum(1 for i in self.instructions if i.i_out == ins.i_out)

            alpha /= x
            alpha = sqrt(alpha)
            normalization_coefficients.append(alpha)

        self.instructions = [
            Instruction(
                ins.i_in1,
                ins.i_in2,
                ins.i_out,
                ins.connection_mode,
                ins.has_weight,
                alpha,
                ins.path_shape,
            )
            for ins, alpha in zip(self.instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions if ins.has_weight
        )
        self.internal_weights = internal_weights

        if internal_weights and self.weight_numel > 0:
            self.weight = torch.nn.Parameter(
                torch.randn(self.weight_numel, dtype=torch.float32)
            )
        else:
            self.register_buffer("weight", torch.empty(0, dtype=torch.float32))

        graphmod = _codegen_graph(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
        )
        self._codegen_register({"_compiled_main_left_right": graphmod})

    def _get_instruction(
        self,
        input1,
        input2,
        output,
        learnable_weight=True,
        connection_mode="uvu",
        reduce_sameorder=True,
    ):
        input1 = o3.Irreps(input1)
        input2 = o3.Irreps(input2)
        output = o3.Irreps(output)

        if not learnable_weight:
            connection_mode = "uvu"

        irreps_output = []
        instructions = []

        for i, (mul, ir_in) in enumerate(input1):
            for j, (_, ir_edge) in enumerate(input2):
                for ir_out in ir_in * ir_edge:
                    if ir_out in output:
                        k = len(irreps_output)
                        irreps_output.append((mul, ir_out))
                        instructions.append(
                            (i, j, k, connection_mode, learnable_weight)
                        )

        return instructions, o3.Irreps(irreps_output)

    def get_weight_byL1L2L3(self, L1, L2, L3):
        return self.weights_dict[(L1, L2, L3)]

    def forward(self, x, y, weight=None):
        if self.weight.device != x.device:
            self.to(x.device)

        if weight is None:
            weight = self.weight
        elif weight.device != x.device:
            weight = weight.to(x.device)

        return self._compiled_main_left_right(x, y, weight)


class DepthWiseTensorProduct_reducesameorder(Simple_TensorProduct_oTchannel):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        max_ir=None,
        irrep_normalization="none",
        path_normalization="none",
        connection_mode="uvu",
        learnable_weight=True,
        **kwargs,
    ):
        irreps_in1 = (
            o3.Irreps(irreps_in1) if isinstance(irreps_in1, str) else irreps_in1
        )
        irreps_in2 = (
            o3.Irreps(irreps_in2) if isinstance(irreps_in2, str) else irreps_in2
        )

        instr = []
        out_source = []

        if max_ir is None:
            irreps_out = (
                o3.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
            )

            for i_1, (_, ir_1) in enumerate(irreps_in1):
                for i_2, (_, ir_2) in enumerate(irreps_in2):
                    for i_out, (_, ir_out) in enumerate(irreps_out):
                        if ir_out not in ir_1 * ir_2:
                            continue

                        instr.append(
                            (i_1, i_2, i_out, connection_mode, learnable_weight)
                        )
                        out_source.append((ir_1.l, ir_2.l, ir_out.l))

        else:
            last_m1 = None
            last_m2 = None

            for i_1, (m1, ir_1) in enumerate(irreps_in1):
                last_m1 = m1
                for i_2, (m2, ir_2) in enumerate(irreps_in2):
                    last_m2 = m2
                    for ir_out in ir_1 * ir_2:
                        if ir_out.l > max_ir + max(irreps_in1.ls) - ir_2.l:
                            continue

                        instr.append(
                            (
                                i_1,
                                i_2,
                                ir_out.l,
                                connection_mode,
                                learnable_weight,
                            )
                        )
                        out_source.append((ir_1.l, ir_2.l, ir_out.l))

            max_out_order = max([i[2] for i in instr])
            irreps_out = "+".join(
                [
                    "{c}x0e",
                    "{c}x1e",
                    "{c}x2e",
                    "{c}x3e",
                    "{c}x4e",
                    "{c}x5e",
                    "{c}x6e",
                    "{c}x7e",
                    "{c}x8e",
                ][: max_out_order + 1]
            )
            irreps_out = irreps_out.format(c=last_m1 * last_m2)
            irreps_out = o3.Irreps(irreps_out)

        self.out_source = out_source

        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )

        fwi = 0
        self.weights_dict = {}

        for ins in self.instructions:
            mi1 = self.irreps_in1[ins.i_in1]
            mi2 = self.irreps_in2[ins.i_in2]
            mo = self.irreps_out[ins.i_out]

            if ins.has_weight:
                self.weights_dict[(mi1.ir.l, mi2.ir.l, mo.ir.l)] = slice(
                    fwi,
                    fwi + prod(ins.path_shape),
                )
                fwi += prod(ins.path_shape)


class DepthwiseTensorProduct_wosort(Simple_TensorProduct_oTchannel):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        filter_ir_out=None,
        max_ir=1000,
        irrep_normalization=None,
        path_normalization=None,
        learnable_weight=False,
        connection_mode="uvu",
        **kwargs,
    ):
        irreps_in1 = o3.Irreps(irreps_in1).simplify()
        irreps_in2 = o3.Irreps(irreps_in2).simplify()

        if filter_ir_out is not None:
            filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]

        out = []
        instr = []
        out_source = []

        max_l_in1 = max(irreps_in1.ls)

        for i_1, (m1, ir_1) in enumerate(irreps_in1):
            for i_2, (_, ir_2) in enumerate(irreps_in2):
                for ir_out in ir_1 * ir_2:
                    if ir_out.l > max_ir + max_l_in1 - ir_2.l:
                        continue

                    i_out = len(out)
                    out.append((m1, ir_out))
                    instr.append((i_1, i_2, i_out, connection_mode, learnable_weight))
                    out_source.append((ir_1.l, ir_2.l, ir_out.l))

        out = o3.Irreps(out)
        self.out_source = out_source

        super().__init__(
            irreps_in1,
            irreps_in2,
            out,
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )


class FullyConnectedTensorProductWigner6j(Simple_TensorProduct_oTchannel):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        rij_order,
        irrep_normalization="none",
        path_normalization="none",
        previous_out_source=None,
        learnable_weight=False,
        connection_mode="uvu",
        simulate_tp=None,
        **kwargs,
    ):
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        self.ins = []
        self.info = []

        for i_1, (_, ir_1) in enumerate(irreps_in1):
            for i_2, (_, ir_2) in enumerate(irreps_in2):
                for i_out, (_, ir_out) in enumerate(irreps_out):
                    if ir_out not in ir_1 * ir_2:
                        continue

                    a, b, d = previous_out_source[i_1]
                    c = ir_2.l
                    abc = ir_out.l

                    if b + c != rij_order:
                        continue

                    bc = b + c
                    coefficient = math.comb(rij_order, b) * (-1) ** b

                    path_weight = coefficient * float(
                        wigner_6j(a, b, d, c, abc, bc)
                        * ((-1) ** (a + b + c + abc))
                        * math.sqrt((2 * d + 1) * (2 * bc + 1))
                    )

                    if path_weight != 0:
                        self.ins.append(
                            (
                                i_1,
                                i_2,
                                i_out,
                                connection_mode,
                                learnable_weight,
                                path_weight,
                            )
                        )
                        self.info.append((a, bc, abc))

        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            self.ins,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            path_weight_sqrt=False,
            **kwargs,
        )

        self.simulate_tp = simulate_tp

        gm = _codegen_graph(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            self.instructions,
            self.simulate_tp,
            self.info,
        )

        assert gm is not None

        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self._codegen_register({"_compiled_main_left_right": gm})

    def forward(self, x, y, weight=None):
        assert x.shape[-2:].numel() == self._in1_dim
        assert y.shape[-2:].numel() == self._in2_dim

        if self.weight.device != x.device:
            self.to(x.device)

        weight = self.simulate_tp.weight
        return self._compiled_main_left_right(x, y, weight)


class E2TensorProductArbitraryOrder(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        head,
        order,
        learnable_weight=True,
        connection_mode="uvw",
        path_normalization="element",
    ):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.order = order
        self.head = head

        self.in_c = o3.Irreps(self.irreps_in)[0][0]
        self.out_c = o3.Irreps(self.irreps_out)[0][0]
        self.lmax = e3nn.o3.Irreps(irreps_in)[-1][1][0]

        assert connection_mode in ["uvw", "uvu"]

        if not learnable_weight:
            connection_mode = "uvu"

        self.tensor_product_tp_component_1 = DepthWiseTensorProduct_reducesameorder(
            irreps_in,
            f"1x{order}e",
            irreps_out,
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
        )

        self.components = nn.ModuleList(
            [
                self._create_component(i, learnable_weight, connection_mode)
                for i in range(1, order + 1)
            ]
        )

        self.coeffs = self.get_coeffs()

        if order > 6:
            raise ValueError("Coeffs for order > 6 not implemented")

        if path_normalization == "element" or path_normalization is None:
            pn = 1 / torch.sqrt(
                get_path_norm(irreps_in, f"1x{order}e", irreps_in).reshape(1, -1, 1)
            )
            self.register_buffer("path_norm", pn)
        else:
            self.register_buffer("path_norm", torch.ones(1))

    def _create_component(self, i, learnable_weight, connection_mode):
        tp_ws = DepthwiseTensorProduct_wosort(
            self.irreps_in,
            o3.Irreps(f"1x{i}e"),
            max_ir=e3nn.o3.Irreps(self.irreps_in)[-1][1].l + (self.order - i),
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=False,
        )

        w6j_tp = FullyConnectedTensorProductWigner6j(
            tp_ws.irreps_out,
            o3.Irreps(f"1x{self.order - i}e"),
            self.irreps_out,
            rij_order=self.order,
            previous_out_source=tp_ws.out_source,
            irrep_normalization="component",
            path_normalization="none",
            learnable_weight=learnable_weight,
            connection_mode=connection_mode,
            simulate_tp=self.tensor_product_tp_component_1,
        )

        return nn.ModuleDict(
            {
                "tp_without_sort": tp_ws,
                "wigner_6j_tp": w6j_tp,
            }
        )

    @staticmethod
    def get_coeffs():
        return (
            1.0,
            2.046653509140,
            1.29441716,
            0.84739512,
            0.56493002,
            0.38087577,
            0.25875416,
        )

    def _build_y_powers(self, pos):
        y_powers = [
            self.coeffs[0]
            * torch.ones(
                pos.shape[:-1] + (1, 1),
                dtype=pos.dtype,
                device=pos.device,
            )
        ]

        for i in range(1, self.order + 1):
            y = e3nn.o3.spherical_harmonics(
                i,
                pos,
                normalize=False,
                normalization="integral",
            ).unsqueeze(-1)

            y_powers.append(self.coeffs[i] * y)

        return y_powers

    def _reduce_by_alpha(self, src, alpha_ij, f_sparse_idx_expnode):
        """
        src: [N2, O, C]
        alpha_ij: [N1, K, Head]
        f_sparse_idx_expnode: [N1, K] or None

        returns: [N1, O, C]
        """
        n1 = alpha_ij.shape[0]
        head = self.head

        n2 = src.shape[0]
        o = src.shape[-2]
        c = src.shape[-1]
        inner = c // head

        src = src.reshape(n2, o, head, inner)

        if f_sparse_idx_expnode is not None:
            k = f_sparse_idx_expnode.shape[1]

            gathered = src.index_select(
                0,
                f_sparse_idx_expnode.reshape(-1),
            ).reshape(n1, k, o, head, inner)

            out = (gathered * alpha_ij.unsqueeze(2).unsqueeze(-1)).sum(dim=1)

            return out.reshape(n1, o, c)

        # Dense fallback path.
        src = src.permute(2, 0, 1, 3).reshape(head, n2, o * inner)
        alpha = alpha_ij.permute(2, 0, 1)

        out = torch.matmul(alpha, src)
        out = out.reshape(head, n1, o, inner).permute(1, 2, 0, 3)

        return out.reshape(n1, o, c)

    def forward(
        self,
        pos,
        exp_pos,
        h,
        exp_h,
        alpha_ij,
        f_sparse_idx_expnode=None,
        batched_data=None,
    ):
        if self.path_norm.device != pos.device:
            self.to(pos.device)

        alpha_ij = alpha_ij.contiguous()

        if f_sparse_idx_expnode is not None:
            f_sparse_idx_expnode = f_sparse_idx_expnode.contiguous()

        if batched_data is not None and "Y_powers" in batched_data:
            Y_powers = batched_data["Y_powers"]
            exp_Y_powers = batched_data["exp_Y_powers"]
        else:
            Y_powers = self._build_y_powers(pos)
            exp_Y_powers = self._build_y_powers(exp_pos)

        component_1 = self._reduce_by_alpha(
            exp_h,
            alpha_ij,
            f_sparse_idx_expnode,
        )

        component_1 = self.tensor_product_tp_component_1(
            component_1,
            Y_powers[self.order],
        )

        out = component_1

        for i, component in enumerate(self.components):
            k = i + 1

            c = component["tp_without_sort"](
                exp_h,
                exp_Y_powers[k],
            )

            c = self._reduce_by_alpha(
                c,
                alpha_ij,
                f_sparse_idx_expnode,
            )

            c = component["wigner_6j_tp"](
                c,
                Y_powers[self.order - k],
            )

            out = out + c

        return out * self.path_norm


# ==============================================================================
# Model Competition Interface
# ==============================================================================


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        head = 64
        hidden = 1
        order = 2

        irreps_in = "+".join(
            [
                f"{head * hidden}x0e",
                f"{head * hidden}x1e",
                f"{head * hidden}x2e",
                f"{head * hidden}x3e",
            ]
        )

        irreps_out = "512x0e+512x1e+512x2e+512x3e"

        self.model = E2TensorProductArbitraryOrder(
            irreps_in,
            irreps_out,
            head,
            order=order,
            learnable_weight=True,
            connection_mode="uvw",
            path_normalization="element",
        )

    def forward(self, pos, exp_pos, exp_h, alpha_ij, f_sparse_idx_expnode):
        return self.model(
            pos,
            exp_pos,
            None,
            exp_h,
            alpha_ij,
            f_sparse_idx_expnode,
            batched_data=None,
        )


def get_inputs():
    N1 = 2186
    N2 = 6473
    K = 20
    Head = 64
    L_max = 3
    In_Channels = 64

    dtype = torch.float32
    device = "npu:0"

    pos = torch.randn(N1, 3, dtype=dtype, device=device)
    exp_pos = torch.randn(N2, 3, dtype=dtype, device=device)
    exp_h = torch.randn(N2, (L_max + 1) ** 2, In_Channels, dtype=dtype, device=device)
    alpha_ij = torch.randn(N1, K, Head, dtype=dtype, device=device)
    f_sparse_idx_expnode = torch.randint(
        0,
        N2,
        (N1, K),
        dtype=torch.int64,
        device=device,
    )

    return [pos, exp_pos, exp_h, alpha_ij, f_sparse_idx_expnode]


def get_init_inputs():
    return []

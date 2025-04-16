# Copyright (c) 2025, DeepLink.
from dlblas.autotune.configs import AutotuneConfig
from dlblas.autotune.space import ChoiceSpace, DictSpace, DiscreteSpace, FixedSpace, PowerOfTwoSpace, RangeSapce
from dlblas.op_registry import op_registry
from dlblas.op_struct import OpImpl, OpParams, Tensor
from dlblas.symbolic_var import SymVar

register_dlblas_op = op_registry.register


def get_list_op_names() -> [str]:
    return op_registry.get_list_op_names()


def get_args_from_op_name(name: str):
    return op_registry.get_args_from_op_name(name)


def get_op(name: str, args):
    '''based on name and args,
    return OpImpl
    '''
    return op_registry.get_op(name, args)

from typing import Optional

from dlblas.kernels.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction


def liger_fused_linear_cross_entropy(
    input,
    weight,
    target,
    bias=None,
    ce_weight=None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = 'mean',
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
):
    loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
        input,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
    )
    if not return_z_loss:
        return loss
    return loss, z_loss

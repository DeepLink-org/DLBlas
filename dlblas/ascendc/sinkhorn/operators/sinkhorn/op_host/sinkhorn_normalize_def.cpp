/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * @file sinkhorn_normalize_def.cpp
 * @brief SinkhornNormalize operator definition
 *
 * Input: 4D tensor [B, S, 4, 4] (float32)
 * Output: 4D tensor [B, S, 4, 4] (float32), doubly stochastic per 4x4 sub-matrix
 */

#include "register/op_def_registry.h"

namespace ops {
class SinkhornNormalize : public OpDef {
public:
    explicit SinkhornNormalize(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("eps")
            .AttrType(OPTIONAL)
            .Float(1e-6f);
        this->Attr("repeat")
            .AttrType(OPTIONAL)
            .Int(10);

        // DAV_2201 (ascend910b) config
        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
    }
};
OP_ADD(SinkhornNormalize);
} // namespace ops

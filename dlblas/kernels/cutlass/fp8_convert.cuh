# Copyright (c) 2025, DeepLink.
#include <cutlass/numeric_conversion.h>

static void convert(cutlass::float_e4m3_t & dest, cutlass::float_e5m2_t & source) {
    cutlass::NumericConverter<cutlass::float_e4m3_t, cutlass::float_e5m2_t, cutlass::FloatRoundStyle::round_toward_zero> converter;
    dest = converter(source);
}

// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/searchsorted.h"

#define instantiate_searchsorted(tname, type)                                \
  instantiate_kernel("searchsorted_left_" #tname, searchsorted_left, type)   \
  instantiate_kernel("searchsorted_right_" #tname, searchsorted_right, type)

instantiate_searchsorted(float32, float)
instantiate_searchsorted(float16, half)
instantiate_searchsorted(bfloat16, bfloat16_t)
instantiate_searchsorted(int32, int32_t)
instantiate_searchsorted(uint32, uint32_t)
instantiate_searchsorted(int16, int16_t)
instantiate_searchsorted(uint16, uint16_t)
instantiate_searchsorted(int8, int8_t)
instantiate_searchsorted(uint8, uint8_t)

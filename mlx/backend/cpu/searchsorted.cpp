// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T>
void searchsorted_impl(
    const T* sorted,
    const T* values,
    int32_t* out,
    int n,
    int m,
    bool right) {
  for (int i = 0; i < m; i++) {
    T val = values[i];
    int lo = 0, hi = n;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (right ? !(val < sorted[mid]) : sorted[mid] < val) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out[i] = lo;
  }
}

void SearchSorted::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& sorted = inputs[0];
  auto& values = inputs[1];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  int n = sorted.shape(0);
  int m = values.size();

  switch (sorted.dtype()) {
    case float32:
      searchsorted_impl(
          sorted.data<float>(), values.data<float>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case float16:
      searchsorted_impl(
          sorted.data<float16_t>(), values.data<float16_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case bfloat16:
      searchsorted_impl(
          sorted.data<bfloat16_t>(), values.data<bfloat16_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case int32:
      searchsorted_impl(
          sorted.data<int32_t>(), values.data<int32_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case uint32:
      searchsorted_impl(
          sorted.data<uint32_t>(), values.data<uint32_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case int16:
      searchsorted_impl(
          sorted.data<int16_t>(), values.data<int16_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case uint16:
      searchsorted_impl(
          sorted.data<uint16_t>(), values.data<uint16_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case int8:
      searchsorted_impl(
          sorted.data<int8_t>(), values.data<int8_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    case uint8:
      searchsorted_impl(
          sorted.data<uint8_t>(), values.data<uint8_t>(),
          out.data<int32_t>(), n, m, right_);
      break;
    default:
      throw std::runtime_error(
          "[SearchSorted] Unsupported dtype for CPU eval.");
  }
}

} // namespace mlx::core

// Copyright © 2024 Apple Inc.

using namespace metal;

template <typename T>
[[kernel]] void searchsorted_left(
    device const T* sorted     [[buffer(0)]],
    device const T* values     [[buffer(1)]],
    device int32_t* out        [[buffer(2)]],
    constant const int& n      [[buffer(3)]],
    uint tid                   [[thread_position_in_grid]]) {
  T val = values[tid];
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (sorted[mid] < val) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  out[tid] = lo;
}

template <typename T>
[[kernel]] void searchsorted_right(
    device const T* sorted     [[buffer(0)]],
    device const T* values     [[buffer(1)]],
    device int32_t* out        [[buffer(2)]],
    constant const int& n      [[buffer(3)]],
    uint tid                   [[thread_position_in_grid]]) {
  T val = values[tid];
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (!(val < sorted[mid])) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  out[tid] = lo;
}

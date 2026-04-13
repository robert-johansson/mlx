// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void SearchSorted::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& sorted = inputs[0];
  auto& values = inputs[1];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  if (values.size() == 0) return;

  std::string type_name = type_to_name(sorted);
  std::string side = right_ ? "right" : "left";
  std::string kernel_name = "searchsorted_" + side + "_" + type_name;

  auto kernel = get_searchsorted_kernel(d, kernel_name, sorted, right_);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(sorted, 0);
  compute_encoder.set_input_array(values, 1);
  compute_encoder.set_output_array(out, 2);
  int n = sorted.shape(0);
  compute_encoder.set_bytes(n, 3);

  // One thread per value to search
  auto grid = MTL::Size(values.size(), 1, 1);
  auto group = MTL::Size(
      std::min(static_cast<size_t>(values.size()),
               static_cast<size_t>(kernel->maxTotalThreadsPerThreadgroup())),
      1, 1);
  compute_encoder.dispatch_threads(grid, group);
}

} // namespace mlx::core

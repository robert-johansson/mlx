// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

struct FusedKernelBuilder {
  std::string os;
  const std::string& kernel_name;
  const std::vector<array>& inputs;
  const std::vector<array>& outputs;
  const std::vector<array>& tape;
  const std::function<bool(size_t)>& is_constant;

  void build(const char* name, bool contiguous) {
    NodeNamer namer;

    // Function parameters.
    std::vector<std::string> params;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant(i)) {
        continue;
      }
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      params.push_back(
          fmt::format("const {}* {}", dtype_to_cuda_type(x.dtype()), xname));
      if (!is_scalar(x) && !contiguous) {
        params.push_back(
            fmt::format(
                "const __grid_constant__ cuda::std::array<int64_t, NDIM> {}_strides",
                xname));
      }
    }
    for (const auto& x : outputs) {
      params.push_back(
          fmt::format(
              "{}* {}", dtype_to_cuda_type(x.dtype()), namer.get_name(x)));
    }
    if (!contiguous) {
      params.push_back(
          "const __grid_constant__ cuda::std::array<int32_t, NDIM> shape");
    }
    params.push_back("IdxT size");

    // Build function signature.
    if (contiguous) {
      os += "template <typename IdxT = uint32_t, int work_per_thread = 1>\n";
    } else {
      os +=
          "template <int NDIM, typename IdxT = uint32_t, int work_per_thread = 1>\n";
    }
    os += fmt::format("__global__ void {}(\n", kernel_name + name);
    for (size_t i = 0; i < params.size(); ++i) {
      os += "    ";
      os += params[i];
      if (i != params.size() - 1) {
        os += ",\n";
      }
    }
    os += ") {\n";

    // Index. For non contiguous kernels we create a separate index
    // variable per variable otherwise everyone uses `index`.
    os +=
        "  IdxT index = cg::this_grid().thread_rank() * work_per_thread;\n"
        "  if (index >= size) {\n"
        "    return;\n"
        "  }\n";
    if (!contiguous) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "  IdxT " + xname + "_idx = 0;\n";
      }
      os += "  {\n";
      os += "    IdxT loc = index;\n";
      os +=
          "    #pragma unroll\n"
          "    for (int i = NDIM - 1; i >= 0; i--) {\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += "      " + xname + "_idx += (loc \% shape[i]) * IdxT(" + xname +
            "_strides[i]);\n";
      }
      os +=
          "      loc /= shape[i];\n"
          "    }\n"
          "  }\n";
    }

    // Vectorized read loop
    if (contiguous) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        const std::string& xname = namer.get_name(x);
        std::string type = dtype_to_cuda_type(x.dtype());
        os += fmt::format(
            "  auto vec_{0} = load_vector<work_per_thread, {1}>({0} + index, 0, size - index, 0);\n",
            xname,
            type);
      }
    }

    // Create some space for the outputs
    for (const auto& x : outputs) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      os += fmt::format(
          "  AlignedVector<{}, work_per_thread> vec_{};\n", type, xname);
    }

    // Work loop
    if (!contiguous) {
      os +=
          "\n"
          "  for (int i = 0; i < work_per_thread && index < size; i++) {\n";
    } else {
      os +=
          "\n"
          "  #pragma unroll\n"
          "  for (int i = 0; i < work_per_thread; i++) {\n";
    }

    // Read inputs.
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      std::string value;
      if (is_constant(i)) {
        std::ostringstream ss;
        print_constant(ss, x);
        value = fmt::format("static_cast<{}>({})", type, ss.str());
      } else if (is_scalar(x)) {
        value = fmt::format("{}[0]", xname);
      } else if (contiguous) {
        value = fmt::format("vec_{}[i]", xname);
      } else {
        value = fmt::format("{}[{}_idx]", xname, xname);
      }
      os += fmt::format("    {} tmp_{} = {};\n", type, xname, value);
    }

    // Write tape.
    for (const auto& x : tape) {
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      std::string value;
      if (is_static_cast(x.primitive())) {
        value = fmt::format(
            "static_cast<{}>(tmp_{})", type, namer.get_name(x.inputs()[0]));
      } else {
        value = x.primitive().name();
        value += "{}(";
        for (size_t i = 0; i < x.inputs().size() - 1; ++i) {
          value += fmt::format("tmp_{}, ", namer.get_name(x.inputs()[i]));
        }
        value += fmt::format("tmp_{})", namer.get_name(x.inputs().back()));
      }
      os += fmt::format("    {} tmp_{} = {};\n", type, xname, value);
    }

    // Write output.
    for (const auto& x : outputs) {
      os += fmt::format("    vec_{0}[i] = tmp_{0};\n", namer.get_name(x));
    }

    // End of work loop
    if (!contiguous) {
      os += "\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& x = inputs[i];
        const std::string& xname = namer.get_name(x);
        if (is_scalar(x) || is_constant(i)) {
          continue;
        }
        os += fmt::format("    {0}_idx += {0}_strides[NDIM - 1];\n", xname);
      }
    }
    os += "  }\n";

    // Store the output to global memory
    for (const auto& x : outputs) {
      os += fmt::format(
          "  store_vector({0} + index, 0, vec_{0}, size - index);\n",
          namer.get_name(x));
    }

    os += "}\n";
  }
};

namespace {

// Fused reduce codegen (genmlx-7dm0): the tape ends with a float32 Sum over
// a 1-D (full) or 2-D (last-axis) domain — admitted by is_reduce_fusion_root
// in compile.cpp. One block per output row; threads stride the row
// computing the fused elementwise producer inline and accumulating; block
// reduction via cooperative groups. Every input is indexed with a
// per-input (row, col) stride pair, so broadcast/strided layouts need no
// materializing copies.
struct FusedReduceKernelBuilder {
  std::string os;
  const std::string& kernel_name;
  const std::vector<array>& inputs;
  const std::vector<array>& tape;
  const std::function<bool(size_t)>& is_constant;

  void build() {
    NodeNamer namer;
    const array& red = tape.back();
    const array& red_in = red.inputs()[0];

    std::vector<std::string> params;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant(i)) {
        continue;
      }
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      params.push_back(
          fmt::format("const {}* {}", dtype_to_cuda_type(x.dtype()), xname));
      if (!is_scalar(x)) {
        params.push_back(
            fmt::format(
                "const __grid_constant__ cuda::std::array<int64_t, 2> {}_strides",
                xname));
      }
    }
    params.push_back("float* red_out");
    params.push_back("int64_t n_rows");
    params.push_back("int64_t row_size");

    os += fmt::format("__global__ void {}_rowsum(\n", kernel_name);
    for (size_t i = 0; i < params.size(); ++i) {
      os += "    ";
      os += params[i];
      if (i != params.size() - 1) {
        os += ",\n";
      }
    }
    os += ") {\n";
    os +=
        "  int64_t row = blockIdx.x;\n"
        "  if (row >= n_rows) {\n"
        "    return;\n"
        "  }\n"
        "  float acc = 0.0f;\n"
        "  for (int64_t col = threadIdx.x; col < row_size; col += blockDim.x) {\n";

    // Read inputs at (row, col).
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& x = inputs[i];
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      std::string value;
      if (is_constant(i)) {
        std::ostringstream ss;
        print_constant(ss, x);
        value = fmt::format("static_cast<{}>({})", type, ss.str());
      } else if (is_scalar(x)) {
        value = fmt::format("{}[0]", xname);
      } else {
        value = fmt::format(
            "{0}[row * {0}_strides[0] + col * {0}_strides[1]]", xname);
      }
      os += fmt::format("    {} tmp_{} = {};\n", type, xname, value);
    }

    // Tape (excluding the trailing Reduce).
    for (size_t t = 0; t + 1 < tape.size(); ++t) {
      const auto& x = tape[t];
      const std::string& xname = namer.get_name(x);
      std::string type = dtype_to_cuda_type(x.dtype());
      std::string value;
      if (is_static_cast(x.primitive())) {
        value = fmt::format(
            "static_cast<{}>(tmp_{})", type, namer.get_name(x.inputs()[0]));
      } else {
        value = x.primitive().name();
        value += "{}(";
        for (size_t i = 0; i < x.inputs().size() - 1; ++i) {
          value += fmt::format("tmp_{}, ", namer.get_name(x.inputs()[i]));
        }
        value += fmt::format("tmp_{})", namer.get_name(x.inputs().back()));
      }
      os += fmt::format("    {} tmp_{} = {};\n", type, xname, value);
    }

    os += fmt::format(
        "    acc += static_cast<float>(tmp_{});\n", namer.get_name(red_in));
    os += "  }\n";

    // Block reduction: warp shuffles + one shared staging round.
    os +=
        "  __shared__ float warp_sums[32];\n"
        "  auto warp = cg::tiled_partition<32>(cg::this_thread_block());\n"
        "  acc = cg::reduce(warp, acc, cg::plus<float>());\n"
        "  if (warp.thread_rank() == 0) {\n"
        "    warp_sums[warp.meta_group_rank()] = acc;\n"
        "  }\n"
        "  __syncthreads();\n"
        "  if (warp.meta_group_rank() == 0) {\n"
        "    int nwarps = (blockDim.x + 31) / 32;\n"
        "    float v = (warp.thread_rank() < nwarps)\n"
        "        ? warp_sums[warp.thread_rank()] : 0.0f;\n"
        "    v = cg::reduce(warp, v, cg::plus<float>());\n"
        "    if (warp.thread_rank() == 0) {\n"
        "      red_out[row] = v;\n"
        "    }\n"
        "  }\n"
        "}\n";
  }
};

} // namespace

} // namespace cu

constexpr const char* g_jit_includes = R"(
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/ternary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>
)";

constexpr const char* g_jit_reduce_includes = R"(
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/ternary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
)";

// Eval path for reduce-rooted Compiled tapes (genmlx-7dm0): the tape's
// last entry is a float32 Sum whose input domain is 1-D (full sum) or 2-D
// (last-axis sum). One fused kernel: block per row, strided reads per
// input, cg block reduction.
static void compiled_eval_gpu_reduce(
    const Stream& s,
    const std::string& lib_name,
    const std::vector<array>& compiled_inputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& encoder = cu::get_command_encoder(s);

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), lib_name, [&]() {
    cu::FusedReduceKernelBuilder builder{
        g_jit_reduce_includes, lib_name, compiled_inputs, tape, is_constant};
    builder.os +=
        "namespace mlx::core::cu {\n\n"
        "namespace cg = cooperative_groups;\n\n";
    builder.build();
    builder.os += "\n} // namespace mlx::core::cu\n";
    std::vector<std::string> kernel_names;
    kernel_names.push_back(fmt::format("mlx::core::cu::{}_rowsum", lib_name));
    return std::make_tuple(
        false, std::move(builder.os), std::move(kernel_names));
  });

  // Row geometry from the (shaped-compile) trace domain.
  const array& red_in = tape.back().inputs()[0];
  const auto& dshape = red_in.shape();
  int64_t n_rows = dshape.size() == 2 ? dshape[0] : 1;
  int64_t row_size = dshape.back();

  cu::KernelArgs args;
  // Right-align each runtime input's strides to the domain: stride 0 for
  // broadcast (size-1 or missing) dims. Storage must stay alive through
  // the launch and must not reallocate.
  std::vector<std::array<int64_t, 2>> stride_storage;
  stride_storage.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) {
      continue;
    }
    const auto& x = inputs[i];
    args.append(x);
    if (!is_scalar(x)) {
      int64_t rs = 0;
      int64_t cs = 0;
      int xnd = x.ndim();
      if (xnd >= 1 && x.shape(xnd - 1) > 1) {
        cs = x.strides()[xnd - 1];
      }
      if (dshape.size() == 2 && xnd == 2 && x.shape(0) > 1) {
        rs = x.strides()[0];
      }
      stride_storage.push_back({rs, cs});
      args.append_ptr(stride_storage.back().data());
    }
  }

  outputs[0].set_data(cu::malloc_async(outputs[0].nbytes(), encoder));
  args.append(outputs[0]);
  args.append<int64_t>(n_rows);
  args.append<int64_t>(row_size);

  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  encoder.set_output_array(outputs[0]);

  if (outputs[0].size() == 0) {
    return;
  }

  std::string kernel_name =
      fmt::format("mlx::core::cu::{}_rowsum", lib_name);
  auto [kernel, max_block_dims] = mod.get_kernel_and_dims(kernel_name);
  uint32_t threads = 32;
  while (threads < 256 && static_cast<int64_t>(threads) < row_size) {
    threads *= 2;
  }
  dim3 grid(static_cast<unsigned int>(n_rows), 1, 1);
  dim3 block(threads, 1, 1);
  encoder.add_kernel_node_raw(kernel, grid, block, {}, 0, args.args());
}

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Compiled::eval_gpu");
  auto& s = stream();

  // Reduce-rooted tape → the fused reduce path (genmlx-7dm0).
  if (!tape_.empty() && tape_.back().has_primitive() &&
      typeid(tape_.back().primitive()) == typeid(Reduce)) {
    compiled_eval_gpu_reduce(
        s, lib_name(), inputs_, tape_, is_constant_, inputs, outputs);
    return;
  }

  // Determine the work per thread for the vectorized reads/writes. We take it
  // as 16 over the max itemsize for the outputs. Another heuristic could be
  // over the max itemsize of all arrays.
  int max_size = 1;
  for (const auto& x : outputs) {
    max_size = (max_size > x.itemsize()) ? max_size : x.itemsize();
  }
  int work_per_thread = 16 / max_size;

  auto& encoder = cu::get_command_encoder(s);

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), lib_name(), [&]() {
    // Build source code.
    cu::FusedKernelBuilder builder{
        g_jit_includes, lib_name(), inputs_, outputs_, tape_, is_constant_};
    builder.os +=
        "namespace mlx::core::cu {\n\n"
        "namespace cg = cooperative_groups;\n\n";
    builder.build("_contiguous", true);
    builder.os += "\n";
    builder.build("_strided", false);
    builder.os += "\n} // namespace mlx::core::cu\n";
    // Build kernel names.
    std::vector<std::string> kernel_names;
    kernel_names.push_back(
        fmt::format(
            "mlx::core::cu::{}_contiguous<uint32_t, {}>",
            lib_name(),
            work_per_thread));
    kernel_names.push_back(
        fmt::format(
            "mlx::core::cu::{}_contiguous<int64_t, {}>",
            lib_name(),
            work_per_thread));
    for (int wpt : {1, work_per_thread}) {
      for (int i = 1; i <= MAX_NDIM; ++i) {
        kernel_names.push_back(
            fmt::format(
                "mlx::core::cu::{}_strided<{}, uint32_t, {}>",
                lib_name(),
                i,
                wpt));
        kernel_names.push_back(
            fmt::format(
                "mlx::core::cu::{}_strided<{}, int64_t, {}>",
                lib_name(),
                i,
                wpt));
      }
    }

    return std::make_tuple(
        false, std::move(builder.os), std::move(kernel_names));
  });

  // Collapse contiguous dims to route to a faster kernel if possible. Also
  // handle all broadcasting.
  auto [contiguous, negative_strides, shape, strides_vec] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Whether to use large index (also true for negative strides).
  bool large =
      negative_strides || compiled_use_large_index(inputs, outputs, contiguous);

  cu::KernelArgs args;
  // Put inputs.
  int strides_index = 1;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    const auto& x = inputs[i];
    args.append(x);
    if (!contiguous && !is_scalar(x)) {
      args.append_ptr(strides_vec[strides_index++].data());
    }
  }

  // Put outputs.
  compiled_allocate_outputs(
      inputs, outputs, is_constant_, contiguous, [&](auto n) {
        return cu::malloc_async(n, encoder);
      });
  for (auto& x : outputs) {
    args.append(x);
  }

  // Put shape and size.
  if (!contiguous) {
    args.append_ptr(shape.data());
  }
  if (large) {
    args.append<int64_t>(outputs[0].data_size());
  } else {
    args.append<uint32_t>(outputs[0].data_size());
  }

  // Choose work per thread
  if (!contiguous && shape.back() % work_per_thread != 0) {
    work_per_thread = 1;
  }

  // Launch kernel.
  const char* index_type = large ? "int64_t" : "uint32_t";
  std::string kernel_name = fmt::format("mlx::core::cu::{}", lib_name());
  if (contiguous) {
    kernel_name +=
        fmt::format("_contiguous<{}, {}>", index_type, work_per_thread);
  } else {
    kernel_name += fmt::format(
        "_strided<{}, {}, {}>", shape.size(), index_type, work_per_thread);
  }
  for (const auto& in : inputs) {
    encoder.set_input_array(in);
  }
  for (const auto& out : outputs) {
    encoder.set_output_array(out);
  }

  auto [kernel, max_block_dims] = mod.get_kernel_and_dims(kernel_name);
  auto [num_blocks, block_dims] =
      get_launch_args(outputs[0], large, work_per_thread, max_block_dims);
  encoder.add_kernel_node_raw(
      kernel, num_blocks, block_dims, {}, 0, args.args());
}

} // namespace mlx::core

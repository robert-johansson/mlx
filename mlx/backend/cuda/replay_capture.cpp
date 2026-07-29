// Copyright © 2026 Apple Inc.
//
// CUDA implementation of the replay-capture sink (genmlx-7prh). See
// mlx/backend/gpu/replay_capture.h for the API contract and
// device.cpp/device.h for the three encoder hook points (per-commit exec
// clone, temporary retention, fallback invalidation).

#include "mlx/backend/gpu/replay_capture.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/utils.h"

#include <pthread.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace mlx::core {

namespace cu {

// Fast-path flag consulted inline by CommandEncoder (device.h): one relaxed
// atomic load per op when no window is open.
std::atomic<bool> g_replay_sink_active{false};

namespace {

struct ReplaySink {
  // Dedicated stream for replay launches, input writes, and output copies.
  // Owned here so replays are self-contained: ordering against the rest of
  // MLX is by "inputs available before copy_into" + "sync before results
  // escape", never by sharing an encoder stream.
  CudaStream stream;
  std::vector<CudaGraphExec> execs;
  // Op-internal temporaries allocated by encoders during the captured eval:
  // their buffers are baked into the exec's kernel params, so they must
  // outlive the sink even though the eval machinery frees its own refs.
  std::vector<std::shared_ptr<array::Data>> temporaries;
  bool valid{true};
  std::string invalid_reason;
  // Foreign-work tripwire: all ops of the single captured eval run on ONE
  // scheduler thread; a commit from a second thread means concurrent GPU
  // work leaked into the window.
  pthread_t commit_thread{};
  bool saw_commit{false};

  explicit ReplaySink(Device& d) : stream(d) {}
};

std::mutex g_sink_mutex;
ReplaySink* g_active_sink{nullptr};

bool replay_debug() {
  static bool dbg = std::getenv("MLX_REPLAY_DEBUG") != nullptr;
  return dbg;
}

Device& sink_device() {
  return cu::device(0);
}

void invalidate_locked(ReplaySink* sink, const char* reason) {
  if (sink->valid) {
    sink->valid = false;
    sink->invalid_reason = reason ? reason : "unknown";
  }
}

} // namespace

// ---- Hooks called from CommandEncoder (device.cpp) ------------------------

void replay_sink_on_commit(cudaGraph_t graph) {
  std::lock_guard<std::mutex> lock(g_sink_mutex);
  auto* sink = g_active_sink;
  if (!sink || !sink->valid) {
    return;
  }
  if (!sink->saw_commit) {
    sink->saw_commit = true;
    sink->commit_thread = pthread_self();
  } else if (!pthread_equal(sink->commit_thread, pthread_self())) {
    invalidate_locked(
        sink, "commits from multiple threads during the capture window");
    return;
  }
  try {
    CudaGraphExec exec;
    exec.instantiate(graph);
    sink->execs.push_back(std::move(exec));
    if (replay_debug()) {
      size_t num_nodes = 0;
      cudaGraphGetNodes(graph, nullptr, &num_nodes);
      fprintf(
          stderr,
          "[replay] sink commit #%zu: %zu nodes\n",
          sink->execs.size(),
          num_nodes);
      std::vector<cudaGraphNode_t> nodes(num_nodes);
      cudaGraphGetNodes(graph, nodes.data(), &num_nodes);
      for (size_t ni = 0; ni < num_nodes; ni++) {
        cudaGraphNodeType type;
        cudaGraphNodeGetType(nodes[ni], &type);
        if (type == cudaGraphNodeTypeKernel) {
          CUDA_KERNEL_NODE_PARAMS p;
          if (cuGraphKernelNodeGetParams(nodes[ni], &p) == CUDA_SUCCESS &&
              p.kernelParams) {
            fprintf(
                stderr,
                "[replay]   node %zu kernel args: %p %p %p\n",
                ni,
                p.kernelParams[0] ? *(void**)p.kernelParams[0] : nullptr,
                p.kernelParams[1] ? *(void**)p.kernelParams[1] : nullptr,
                p.kernelParams[2] ? *(void**)p.kernelParams[2] : nullptr);
          } else {
            fprintf(stderr, "[replay]   node %zu kernel (params unreadable)\n", ni);
          }
        } else {
          fprintf(stderr, "[replay]   node %zu type=%d\n", ni, (int)type);
        }
      }
    }
  } catch (const std::exception& e) {
    (void)cudaGetLastError();
    invalidate_locked(sink, "sink exec instantiation failed");
  }
}

void replay_sink_add_temporary(const std::shared_ptr<array::Data>& data) {
  std::lock_guard<std::mutex> lock(g_sink_mutex);
  if (g_active_sink && g_active_sink->valid) {
    g_active_sink->temporaries.push_back(data);
  }
}

void replay_sink_invalidate(const char* reason) {
  std::lock_guard<std::mutex> lock(g_sink_mutex);
  if (g_active_sink) {
    invalidate_locked(g_active_sink, reason);
  }
}

} // namespace cu

// ---- Public backend-neutral API -------------------------------------------

namespace gpu {

bool replay_capture_supported() {
  return env::get_var("MLX_USE_CUDA_GRAPHS", true);
}

void* replay_capture_begin() {
  if (!replay_capture_supported()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(cu::g_sink_mutex);
  if (cu::g_active_sink) {
    return nullptr; // one window at a time
  }
  cu::sink_device().make_current();
  auto* sink = new cu::ReplaySink(cu::sink_device());
  cu::g_active_sink = sink;
  cu::g_replay_sink_active.store(true, std::memory_order_release);
  return sink;
}

bool replay_capture_end(void* handle, std::string* why) {
  auto* sink = static_cast<cu::ReplaySink*>(handle);
  {
    std::lock_guard<std::mutex> lock(cu::g_sink_mutex);
    if (cu::g_active_sink == sink) {
      cu::g_replay_sink_active.store(false, std::memory_order_release);
      cu::g_active_sink = nullptr;
    }
  }
  if (!sink->valid && why) {
    *why = sink->invalid_reason;
  }
  return sink->valid;
}

size_t replay_capture_graph_count(void* handle) {
  return static_cast<cu::ReplaySink*>(handle)->execs.size();
}

void replay_capture_launch(void* handle) {
  auto* sink = static_cast<cu::ReplaySink*>(handle);
  cu::sink_device().make_current();
  for (auto& exec : sink->execs) {
    CHECK_CUDA_ERROR(cudaGraphLaunch(exec, sink->stream));
  }
  if (cu::replay_debug()) {
    fprintf(stderr, "[replay] launched %zu execs\n", sink->execs.size());
  }
}

std::optional<array> replay_capture_clone_array(
    void* handle,
    const array& src) {
  auto* sink = static_cast<cu::ReplaySink*>(handle);
  // Row-contiguous offset views are staged device-side too (genmlx-qkx6):
  // gpu_ptr adds the offset, and a row-contiguous view's data_size span
  // starting there is exactly its logical contents. Declining them sent
  // slice-backed captured-call inputs to the full eager path (~187 ms per
  // 4000-op chunk on the chunked-HMC shapes).
  if (src.offset() != 0 && !src.flags().row_contiguous) {
    return std::nullopt;
  }
  cu::sink_device().make_current();
  size_t bytes = src.data_size() * src.itemsize();
  array dst(src.shape(), src.dtype(), nullptr, {});
  dst.set_data(
      allocator::malloc(bytes),
      src.data_size(),
      src.strides(),
      src.flags(),
      allocator::free);
  // gpu_ptr, NEVER array::data<T>() here: data() goes through
  // Buffer::raw_ptr(), which MIGRATES a device-pool buffer to unified
  // memory and frees the device buffer — but the captured execs have the
  // ORIGINAL device address frozen in their kernel params. One data() call
  // on a retained array would silently detach every future replay from the
  // buffer we read (the genmlx-7prh stale-replay bug, found by exactly
  // that: the first output clone migrated the output buffer).
  if (bytes > 0) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        gpu_ptr<void>(dst),
        gpu_ptr<void>(src),
        bytes,
        cudaMemcpyDefault,
        sink->stream));
  }
  if (cu::replay_debug()) {
    fprintf(
        stderr,
        "[replay] clone src=%p dst=%p bytes=%zu\n",
        gpu_ptr<void>(src),
        gpu_ptr<void>(dst),
        bytes);
  }
  // "available" is honest only after the sink stream is synced; the caller
  // (the shim's captured-call path) always syncs before a clone escapes.
  dst.set_status(array::Status::available);
  return dst;
}

bool replay_capture_copy_into(void* handle, const array& dst, const array& src) {
  auto* sink = static_cast<cu::ReplaySink*>(handle);
  // src may be a row-contiguous offset view (genmlx-qkx6) — see the note in
  // clone_array. dst is always a sink-owned offset-0 staging clone.
  if ((src.offset() != 0 && !src.flags().row_contiguous) ||
      dst.offset() != 0 || src.data_size() != dst.data_size() ||
      src.dtype() != dst.dtype()) {
    return false;
  }
  cu::sink_device().make_current();
  size_t bytes = src.data_size() * src.itemsize();
  // gpu_ptr for BOTH sides — see the migration note in clone_array above
  // (a data<T>() on the staged dst would orphan the captured input pointer).
  if (bytes > 0) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        gpu_ptr<void>(const_cast<array&>(dst)),
        gpu_ptr<void>(src),
        bytes,
        cudaMemcpyDefault,
        sink->stream));
  }
  if (cu::replay_debug()) {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(sink->stream));
    float host[4] = {0, 0, 0, 0};
    size_t n = bytes < sizeof(host) ? bytes : sizeof(host);
    cudaMemcpy(host, gpu_ptr<void>(const_cast<array&>(dst)), n, cudaMemcpyDefault);
    fprintf(
        stderr,
        "[replay] copy_into dst=%p src=%p bytes=%zu dst[0..3]=%g,%g,%g,%g\n",
        gpu_ptr<void>(const_cast<array&>(dst)),
        gpu_ptr<void>(src),
        bytes,
        host[0],
        host[1],
        host[2],
        host[3]);
  }
  return true;
}

void replay_capture_sync(void* handle) {
  auto* sink = static_cast<cu::ReplaySink*>(handle);
  cu::sink_device().make_current();
  CHECK_CUDA_ERROR(cudaStreamSynchronize(sink->stream));
}

void replay_capture_free(void* handle) {
  auto* sink = static_cast<cu::ReplaySink*>(handle);
  if (!sink) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(cu::g_sink_mutex);
    if (cu::g_active_sink == sink) {
      cu::g_replay_sink_active.store(false, std::memory_order_release);
      cu::g_active_sink = nullptr;
    }
  }
  delete sink;
}

} // namespace gpu

} // namespace mlx::core

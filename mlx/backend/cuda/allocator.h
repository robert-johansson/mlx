// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/backend/cuda/cuda_utils.h"

#include <cuda_runtime.h>
#include <mutex>
#include <set>
#include <utility>

namespace mlx::core::cu {

class CommandEncoder;

using allocator::Buffer;

// Stores cuda-managed unified memory.
struct CudaBuffer {
  void* data;
  size_t size;
  int device; // -1 for managed
};

class SmallSizePool {
 private:
  union Block {
    Block* next;
    CudaBuffer buf;
  };

  Block* buffer_{nullptr};
  void* data_{nullptr};
  Block* next_free_{nullptr};

 public:
  SmallSizePool();
  ~SmallSizePool();

  SmallSizePool(const SmallSizePool&) = delete;
  SmallSizePool& operator=(const SmallSizePool&) = delete;

  CudaBuffer* malloc();
  void free(CudaBuffer* buf);
  bool in_pool(CudaBuffer* buf);
};

class CudaAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  Buffer malloc_async(size_t size, int device, cudaStream_t stream);
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  // Replace the memory of |buf| with unified memory (managed memory or pinned
  // host memory), and copy the data over. Pass |stream| to copy asynchronously.
  void move_to_unified_memory(CudaBuffer& buf, cudaStream_t stream = nullptr);

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

  // Enqueue every pending device-pool free for |device| onto |stream|
  // (stream-ordered after all work already enqueued there). Called by
  // CommandEncoder::commit()/synchronize() on the compute stream — the
  // only point where "after everything enqueued so far" is the same thing
  // as "after every kernel that may still read the buffer". No-op while
  // |stream| is capturing (a captured free node would be replayed).
  void drain_deferred_frees(int device, cudaStream_t stream);

 private:
  void free_cuda_buffer(CudaBuffer* buf);
  void free_async(CudaBuffer& buf, cudaStream_t stream = nullptr);
  // A device-pool free with no known ordering stream. Deferred until a
  // compute-stream drain point instead of cudaFreeAsync on the idle
  // free stream: an unordered free lets the pool reclaim (or unmap at
  // the next trim) memory that an enqueued-but-unexecuted kernel still
  // references — the use-after-free class behind the capped-cache
  // illegal-access crashes (genmlx-q25w).
  void defer_device_free(void* data, size_t size, int device);
  // Emergency ladder when a device allocation fails: evict the buffer
  // cache, synchronize + trim the pool (when legal), retry, and fall
  // back to unified memory before giving up. Returns nullptr only when
  // every stage failed; writes the actual residency (device index, or
  // -1 when the unified fallback was taken) to |out_device|. Called
  // WITHOUT mutex_ held.
  void*
  rescue_device_alloc(size_t size, int device, cudaStream_t stream, int* out_device);

  struct DeferredFree {
    void* data;
    size_t size;
    int device;
  };

  CudaAllocator();
  friend CudaAllocator& allocator();

  std::mutex mutex_;
  size_t memory_limit_;
  size_t free_limit_;
  size_t total_memory_;
  size_t max_pool_size_;
  BufferCache<CudaBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  std::vector<CudaStream> free_streams_;
  std::vector<cudaMemPool_t> mem_pools_;
  SmallSizePool scalar_pool_;
  // Guarded by its own mutex: free_async is reached both with and
  // without mutex_ held (free_cuda_buffer vs move_to_unified_memory),
  // and drain runs from encoder threads.
  std::mutex deferred_mutex_;
  std::vector<DeferredFree> deferred_frees_;
  size_t deferred_bytes_{0};
};

CudaAllocator& allocator();

Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::cu

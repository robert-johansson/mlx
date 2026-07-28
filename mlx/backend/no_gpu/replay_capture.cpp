// Copyright © 2026 Apple Inc.
//
// No-GPU stub for the replay-capture API (genmlx-7prh): CPU-only builds
// have no graph execs to capture. begin() returning nullptr routes callers
// onto the plain compiled-replay path.

#include "mlx/backend/gpu/replay_capture.h"

namespace mlx::core::gpu {

bool replay_capture_supported() {
  return false;
}

void* replay_capture_begin() {
  return nullptr;
}

bool replay_capture_end(void*, std::string* why) {
  if (why) {
    *why = "replay capture is not supported on this backend";
  }
  return false;
}

size_t replay_capture_graph_count(void*) {
  return 0;
}

void replay_capture_launch(void*) {}

std::optional<array> replay_capture_clone_array(void*, const array&) {
  return std::nullopt;
}

bool replay_capture_copy_into(void*, const array&, const array&) {
  return false;
}

void replay_capture_sync(void*) {}

void replay_capture_free(void*) {}

} // namespace mlx::core::gpu

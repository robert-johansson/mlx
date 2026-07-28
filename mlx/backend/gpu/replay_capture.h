// Copyright © 2026 Apple Inc.
//
// Replay capture (genmlx-7prh): capture the eval of a fixed computation
// graph ONCE into retained, instantiated CUDA graph execs, then re-run it
// with launch-only cost. The steady-state contract of mlx::core::compile
// replays still pays a full tape clone plus a per-op scheduler walk (capture
// windows, graph keying, exec update) on EVERY call; for fixed-structure
// hot loops (probabilistic-programming inference sweeps and MCMC chains)
// that host work dominates wall clock by 2-3 orders of magnitude over the
// kernel time. This API lets a caller that OWNS the whole eval window
// (single-threaded, synchronous, no concurrent GPU work) freeze the
// launched graphs and their buffers.
//
// Contract:
//  - begin() opens a process-global capture window (nullptr if unsupported,
//    or if a window is already open). Between begin() and end() the caller
//    must run EXACTLY ONE synchronous mlx::core::eval on this thread and
//    nothing else may touch the GPU: every CommandEncoder commit anywhere
//    in the process is captured into the window's sink. A commit from a
//    second distinct thread invalidates the capture (the foreign-work
//    tripwire); so do graph fallbacks and disabled CUDA graphs.
//  - end() closes the window. Returns false (with *why) when the capture
//    is invalid; the handle must still be freed.
//  - launch() enqueues the retained execs, in commit order, on the sink's
//    own dedicated stream. It does NOT sync; call sync() before reading.
//  - clone_array(h, src): eager device-to-device copy of an AVAILABLE,
//    offset-0 array into a freshly allocated owned buffer, enqueued on the
//    sink stream; the result is a detached available array. Used to stage
//    inputs (whose buffers become the memcpy targets of later calls) and to
//    copy outputs out of the retained buffers (so results never alias the
//    next launch's scratch). Returns std::nullopt when the layout is
//    unsupported.
//  - copy_into(h, dst, src): device-to-device memcpy src -> dst's buffer on
//    the sink stream (both available, same size). The per-call input write.
//  - The caller retains (for the handle's lifetime) every array of the
//    captured tape — the buffers baked into the execs. The sink itself
//    retains the op-internal temporaries the encoders allocated during the
//    captured eval, and the execs.
//
// Only the CUDA backend implements this; Metal and no-gpu builds return
// nullptr from begin() so callers fall back to the plain replay path.

#pragma once

#include <optional>
#include <string>

#include "mlx/array.h"

namespace mlx::core::gpu {

// True when the active backend can capture (CUDA with graphs enabled).
bool replay_capture_supported();

// Open the process-global capture window. nullptr = unsupported or busy.
void* replay_capture_begin();

// Close the window. false = capture invalid (reason in *why when non-null).
bool replay_capture_end(void* handle, std::string* why);

// Number of retained graph execs (0 until a valid end()).
size_t replay_capture_graph_count(void* handle);

// Enqueue the retained execs in commit order on the sink stream (no sync).
void replay_capture_launch(void* handle);

// Eager D2D clone on the sink stream; std::nullopt on unsupported layout.
std::optional<array> replay_capture_clone_array(void* handle, const array& src);

// D2D memcpy src -> dst buffer on the sink stream. false on layout/size
// mismatch.
bool replay_capture_copy_into(void* handle, const array& dst, const array& src);

// Synchronize the sink stream (all launches + copies complete).
void replay_capture_sync(void* handle);

// Free the sink: drops retained execs and temporaries. Safe on any state.
void replay_capture_free(void* handle);

} // namespace mlx::core::gpu

#include "amd_graph.h"
#include <new>
#ifdef COMPATCUDA_ENABLE_HIP
#include "handle_registry.h"
#include "hip_interop.h"
#include <vector>
struct AmdGraph_t {
  hipGraph_t graph{};
  hipGraphExec_t executable{};
  hipStream_t stream{};
  int device{};
  bool active{};
  std::vector<const void *> pins;
};
namespace {
void unpin(AmdGraph g) {
  for (auto p : g->pins)
    --amd_handles::entries.at(p)->pins;
  std::vector<const void *>().swap(g->pins);
}
AmdResult end(AmdGraph g) {
  if (!g->active || amd_handles::capture_thread != std::this_thread::get_id())
    return AMD_ERROR_INVALID_VALUE;
  auto status = hipStreamEndCapture(g->stream, &g->graph);
  g->active = false;
  amd_handles::capturing = false;
  if (status != hipSuccess) {
    unpin(g);
    return AMD_ERROR_UNKNOWN;
  }
  status = hipGraphInstantiate(&g->executable, g->graph, nullptr, nullptr, 0);
  if (status != hipSuccess) {
    hipGraphDestroy(g->graph);
    g->graph = nullptr;
    unpin(g);
    return AMD_ERROR_UNKNOWN;
  }
  return AMD_SUCCESS;
}
} // namespace
AmdResult amdGraphBegin(AmdStream stream, AmdGraph *out) {
  try {

    try {
      std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
      if (!out)
        return AMD_ERROR_INVALID_VALUE;
      *out = nullptr;
      if (!amd_handles::valid(stream))
        return AMD_ERROR_INVALID_HANDLE;
      if (amd_handles::capturing)
        return AMD_ERROR_NOT_READY;
      auto g = amd_handles::track(new (std::nothrow) AmdGraph_t{});
      if (!g)
        return AMD_ERROR_OUT_OF_MEMORY;
      amd_native::stream_device(stream, &g->device);
      amd_native::stream_handle(stream, g->device, &g->stream);
      try {
        for (auto &pair : amd_handles::entries)
          if (pair.second->live &&
              pair.second->type != amd_handles::tag<AmdGraph_t>())
            g->pins.push_back(pair.first);
      } catch (...) {
        amd_handles::retire(g);
        throw;
      }
      amd_native::DeviceGuard guard(g->device);
      if (guard.status ||
          hipStreamBeginCapture(g->stream, hipStreamCaptureModeThreadLocal) !=
              hipSuccess) {
        amd_handles::retire(g);
        return AMD_ERROR_UNKNOWN;
      }
      for (auto p : g->pins)
        ++amd_handles::entries.at(p)->pins;
      g->active = true;
      amd_handles::capturing = true;
      amd_handles::capture_thread = std::this_thread::get_id();
      *out = g;
      return AMD_SUCCESS;
    } catch (const std::bad_alloc &) {
      return AMD_ERROR_OUT_OF_MEMORY;
    } catch (...) {
      return AMD_ERROR_UNKNOWN;
    }

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdGraphEnd(AmdGraph g) {
  try {

    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (!amd_handles::valid(g))
      return AMD_ERROR_INVALID_HANDLE;
    amd_native::DeviceGuard guard(g->device);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    return end(g);

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdGraphLaunch(AmdGraph g, AmdStream stream) {
  try {

    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (!amd_handles::valid(g) || !amd_handles::valid(stream))
      return AMD_ERROR_INVALID_HANDLE;
    if (amd_handles::capturing || !g->executable)
      return AMD_ERROR_NOT_READY;
    hipStream_t native{};
    if (!amd_native::stream_handle(stream, g->device, &native))
      return AMD_ERROR_INVALID_VALUE;
    amd_native::DeviceGuard guard(g->device);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    return hipGraphLaunch(g->executable, native) == hipSuccess
               ? AMD_SUCCESS
               : AMD_ERROR_UNKNOWN;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdGraphDestroy(AmdGraph g) {
  try {

    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (!amd_handles::valid(g))
      return AMD_ERROR_INVALID_HANDLE;
    amd_native::DeviceGuard guard(g->device);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    if (g->active) {
      if (amd_handles::capture_thread != std::this_thread::get_id())
        return AMD_ERROR_INVALID_VALUE;
      // End capture even on cancellation; never leave a stream stuck capturing.
      end(g);
    } else if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (hipDeviceSynchronize() != hipSuccess)
      return AMD_ERROR_UNKNOWN;
    if (g->executable && hipGraphExecDestroy(g->executable) != hipSuccess)
      return AMD_ERROR_UNKNOWN;
    g->executable = nullptr;
    if (g->graph && hipGraphDestroy(g->graph) != hipSuccess)
      return AMD_ERROR_UNKNOWN;
    g->graph = nullptr;
    unpin(g);
    amd_handles::retire(g);
    return AMD_SUCCESS;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
#else
AmdResult amdGraphBegin(AmdStream, AmdGraph *out) {
  try {
    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    return AMD_ERROR_NOT_INITIALIZED;
  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdGraphEnd(AmdGraph) {
  try {
    return AMD_ERROR_NOT_INITIALIZED;
  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdGraphLaunch(AmdGraph, AmdStream) {
  try {
    return AMD_ERROR_NOT_INITIALIZED;
  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdGraphDestroy(AmdGraph) {
  try {
    return AMD_ERROR_NOT_INITIALIZED;
  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
#endif

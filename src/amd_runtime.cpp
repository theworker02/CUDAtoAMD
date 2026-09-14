#include "amd_runtime.h"
#include "handle_registry.h"
#include "hip_interop.h"
#include <cstring>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

struct AmdDevice_t {
  int index;
};
struct AmdStream_t {
  hipStream_t value;
  int device;
};
struct AmdEvent_t {
  hipEvent_t value;
};
struct AmdModule_t {
  hipModule_t value;
  int device;
  AmdFunction functions;
};
struct AmdFunction_t {
  hipFunction_t value;
  AmdModule owner;
  AmdFunction next;
};
struct AmdMemPool_t {
  hipMemPool_t value;
  int device;
  std::mutex mutex;
  std::unordered_map<AmdDeviceAddr, size_t> allocations;
};
bool amd_native::device_index(AmdDevice device, int *index) {
  if (!device || !index)
    return false;
  *index = device->index;
  return true;
}
bool amd_native::stream_device(AmdStream stream, int *index) {
  if (!stream || !index)
    return false;
  *index = stream->device;
  return true;
}
bool amd_native::stream_handle(AmdStream stream, int device,
                               hipStream_t *value) {
  if (!value || (stream && stream->device != device))
    return false;
  *value = stream ? stream->value : nullptr;
  return true;
}
namespace {
class DeviceScope {
  int previous_ = -1;

public:
  hipError_t status;
  explicit DeviceScope(int target) : status(hipGetDevice(&previous_)) {
    if (status == hipSuccess && previous_ != target)
      status = hipSetDevice(target);
  }
  ~DeviceScope() {
    if (previous_ >= 0)
      hipSetDevice(previous_);
  }
};
AmdResult result(hipError_t e) {
  switch (e) {
  case hipSuccess:
    return AMD_SUCCESS;
  case hipErrorInvalidValue:
    return AMD_ERROR_INVALID_VALUE;
  case hipErrorOutOfMemory:
    return AMD_ERROR_OUT_OF_MEMORY;
  case hipErrorNotReady:
    return AMD_ERROR_NOT_READY;
  case hipErrorNotInitialized:
    return AMD_ERROR_NOT_INITIALIZED;
  case hipErrorInvalidDevice:
    return AMD_ERROR_DEVICE_NOT_FOUND;
  case hipErrorFileNotFound:
    return AMD_ERROR_MODULE_LOAD_FAILED;
  default:
    return AMD_ERROR_UNKNOWN;
  }
}
hipStream_t native(AmdStream s) { return s ? s->value : nullptr; }
} // namespace
extern "C" AmdResult amdInit(uint32_t flags) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    return result(hipInit(flags));
  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdGetDeviceCount(int *count) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    return count ? result(hipGetDeviceCount(count)) : AMD_ERROR_INVALID_VALUE;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdGetDevice(AmdDevice *out, int index) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    int count = 0;
    if (result(hipGetDeviceCount(&count)) != AMD_SUCCESS || index < 0 ||
        index >= count)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    *out = amd_handles::track(new (std::nothrow) AmdDevice_t{index});
    return *out ? AMD_SUCCESS : AMD_ERROR_OUT_OF_MEMORY;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdDeviceDestroy(AmdDevice device) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(device))
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(device))
      return AMD_ERROR_INVALID_HANDLE;

    if (!device)
      return AMD_ERROR_INVALID_HANDLE;
    amd_handles::retire(device);
    return AMD_SUCCESS;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdGetDeviceProperties(AmdDevice d, AmdDeviceProps *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(d))
      return AMD_ERROR_INVALID_HANDLE;

    if (!d || !out)
      return AMD_ERROR_INVALID_VALUE;
    hipDeviceProp_t p{};
    auto r = result(hipGetDeviceProperties(&p, d->index));
    if (r)
      return r;
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->name, p.name, sizeof(out->name) - 1);
    std::strncpy(out->gcn_arch, p.gcnArchName, sizeof(out->gcn_arch) - 1);
    out->total_vram_bytes = p.totalGlobalMem;
    out->compute_units = p.multiProcessorCount;
    out->supports_wmma =
        -1; // Unknown: this backend has not queried matrix capabilities.
    out->max_threads_per_block = p.maxThreadsPerBlock;
    out->native_wave_size =
        p.warpSize == 32 ? AMD_WAVE_SIZE_32 : AMD_WAVE_SIZE_64;
    out->is_cdna = (!std::strncmp(p.gcnArchName, "gfx908", 6) ||
                    !std::strncmp(p.gcnArchName, "gfx90a", 6) ||
                    !std::strncmp(p.gcnArchName, "gfx94", 5) ||
                    !std::strncmp(p.gcnArchName, "gfx95", 5))
                       ? 1
                       : 0;
    return AMD_SUCCESS;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemPoolCreate(AmdDevice d, size_t initial,
                                      AmdMemPool *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(d))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    if (!d)
      return AMD_ERROR_INVALID_VALUE;
    DeviceScope scope(d->index);
    if (scope.status)
      return result(scope.status);
    hipMemPoolProps p{};
    p.allocType = hipMemAllocationTypePinned;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = d->index;
    auto *v = amd_handles::track(new (std::nothrow) AmdMemPool_t{});
    if (!v)
      return AMD_ERROR_OUT_OF_MEMORY;
    v->device = d->index;
    auto r = result(hipMemPoolCreate(&v->value, &p));
    if (r) {
      amd_handles::retire(v);
      return r;
    }
    if (initial) {
      uint64_t threshold = initial;
      r = result(hipMemPoolSetAttribute(
          v->value, hipMemPoolAttrReleaseThreshold, &threshold));
      if (r) {
        hipMemPoolDestroy(v->value);
        amd_handles::retire(v);
        return r;
      }
    }
    *out = v;
    return AMD_SUCCESS;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemPoolDestroy(AmdMemPool p) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(p))
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(p))
      return AMD_ERROR_INVALID_HANDLE;

    if (!p)
      return AMD_ERROR_INVALID_HANDLE;
    // Destruction requires exclusive caller ownership; outstanding allocations
    // are rejected, not silently orphaned.
    if (!p->allocations.empty())
      return AMD_ERROR_NOT_READY;
    DeviceScope scope(p->device);
    if (scope.status)
      return result(scope.status);
    auto r = result(hipDeviceSynchronize());
    if (r)
      return r;
    r = result(hipMemPoolDestroy(p->value));
    if (!r) {
      std::unordered_map<AmdDeviceAddr, size_t>().swap(p->allocations);
      amd_handles::retire(p);
    }
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemAllocAsync(AmdDeviceAddr *out, size_t bytes,
                                      AmdMemPool p, AmdStream s) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(p))
      return AMD_ERROR_INVALID_HANDLE;
    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = 0;
    if (!p || !bytes || (s && s->device != p->device))
      return AMD_ERROR_INVALID_VALUE;
    DeviceScope scope(p->device);
    if (scope.status)
      return result(scope.status);
    try {
      std::lock_guard<std::mutex> lock(p->mutex);
      void *v = nullptr;
      auto r = result(hipMallocFromPoolAsync(&v, bytes, p->value, native(s)));
      if (r)
        return r;
      try {
        p->allocations.emplace(reinterpret_cast<AmdDeviceAddr>(v), bytes);
      } catch (...) {
        hipFreeAsync(v, native(s));
        hipStreamSynchronize(native(s));
        throw;
      }
      *out = reinterpret_cast<AmdDeviceAddr>(v);
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
extern "C" AmdResult amdMemFreeAsync(AmdDeviceAddr addr, AmdMemPool p,
                                     AmdStream s) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(p))
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(s))
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(p))
      return AMD_ERROR_INVALID_HANDLE;
    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;

    if (!p || (s && s->device != p->device))
      return AMD_ERROR_INVALID_VALUE;
    DeviceScope scope(p->device);
    if (scope.status)
      return result(scope.status);
    try {
      std::lock_guard<std::mutex> lock(p->mutex);
      auto it = p->allocations.find(addr);
      if (it == p->allocations.end())
        return AMD_ERROR_INVALID_VALUE;
      auto r = result(hipFreeAsync(reinterpret_cast<void *>(addr), native(s)));
      if (!r)
        p->allocations.erase(it);
      return r;
    } catch (...) {
      return AMD_ERROR_UNKNOWN;
    }

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemcpyHtoDAsync(AmdDeviceAddr d, const void *s,
                                        size_t n, AmdStream st) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (st && !amd_handles::valid(st))
      return AMD_ERROR_INVALID_HANDLE;

    return result(
        hipMemcpyAsync((void *)d, s, n, hipMemcpyHostToDevice, native(st)));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemcpyDtoHAsync(void *d, AmdDeviceAddr s, size_t n,
                                        AmdStream st) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (st && !amd_handles::valid(st))
      return AMD_ERROR_INVALID_HANDLE;

    return result(
        hipMemcpyAsync(d, (void *)s, n, hipMemcpyDeviceToHost, native(st)));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemcpyDtoDAsync(AmdDeviceAddr d, AmdDeviceAddr s,
                                        size_t n, AmdStream st) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (st && !amd_handles::valid(st))
      return AMD_ERROR_INVALID_HANDLE;

    return result(hipMemcpyAsync((void *)d, (void *)s, n,
                                 hipMemcpyDeviceToDevice, native(st)));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemHostRegister(void *p, size_t n) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    return result(hipHostRegister(p, n, hipHostRegisterDefault));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdMemHostUnregister(void *p) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    return result(hipHostUnregister(p));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdStreamCreate(AmdDevice d, AmdStream *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(d))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out || !d)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    DeviceScope scope(d->index);
    if (scope.status)
      return result(scope.status);
    auto *v = amd_handles::track(new (std::nothrow) AmdStream_t{});
    if (!v)
      return AMD_ERROR_OUT_OF_MEMORY;
    v->device = d->index;
    auto r = result(hipStreamCreate(&v->value));
    if (r)
      amd_handles::retire(v);
    else
      *out = v;
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdStreamDestroy(AmdStream s) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(s))
      return AMD_ERROR_NOT_READY;

    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;

    if (!s)
      return AMD_ERROR_INVALID_HANDLE;
    auto r = result(hipStreamDestroy(s->value));
    if (!r)
      amd_handles::retire(s);
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdStreamSynchronize(AmdStream s) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;

    return result(hipStreamSynchronize(native(s)));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdStreamQuery(AmdStream s) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;

    return result(hipStreamQuery(native(s)));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdEventCreate(AmdEvent *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    auto *v = amd_handles::track(new (std::nothrow) AmdEvent_t{});
    if (!v)
      return AMD_ERROR_OUT_OF_MEMORY;
    auto r = result(hipEventCreate(&v->value));
    if (r)
      amd_handles::retire(v);
    else
      *out = v;
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdEventCreateOnDevice(AmdDevice device, AmdEvent *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(device))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    if (!device)
      return AMD_ERROR_INVALID_HANDLE;
    DeviceScope scope(device->index);
    if (scope.status)
      return result(scope.status);
    return amdEventCreate(out);

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdEventDestroy(AmdEvent e) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(e))
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(e))
      return AMD_ERROR_INVALID_HANDLE;

    if (!e)
      return AMD_ERROR_INVALID_HANDLE;
    auto r = result(hipEventDestroy(e->value));
    if (!r)
      amd_handles::retire(e);
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdEventRecord(AmdEvent e, AmdStream s) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(e))
      return AMD_ERROR_INVALID_HANDLE;
    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;

    return e ? result(hipEventRecord(e->value, native(s)))
             : AMD_ERROR_INVALID_HANDLE;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdStreamWaitEvent(AmdStream s, AmdEvent e) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;
    if (!amd_handles::valid(e))
      return AMD_ERROR_INVALID_HANDLE;

    return e ? result(hipStreamWaitEvent(native(s), e->value, 0))
             : AMD_ERROR_INVALID_HANDLE;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdEventElapsedTime(float *ms, AmdEvent a, AmdEvent b) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(a))
      return AMD_ERROR_INVALID_HANDLE;
    if (!amd_handles::valid(b))
      return AMD_ERROR_INVALID_HANDLE;

    return ms && a && b ? result(hipEventElapsedTime(ms, a->value, b->value))
                        : AMD_ERROR_INVALID_VALUE;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
// Module functions are borrowed handles owned by their module. Unload
// invalidates them.
namespace {
bool validElf(const unsigned char *p, size_t n) {
  // HIP's load-data API has no size parameter. Validate all ELF table/file
  // ranges before forwarding a caller-owned image; this is validation, not a
  // sandbox.
  if (n < 64 ||
      std::memcmp(p,
                  "\x7f"
                  "ELF",
                  4) ||
      p[4] != 2 || p[5] != 1)
    return false;
  auto u16 = [&](size_t x) {
    uint16_t v;
    std::memcpy(&v, p + x, 2);
    return v;
  };
  auto u64 = [&](size_t x) {
    uint64_t v;
    std::memcpy(&v, p + x, 8);
    return v;
  };
  if (u16(18) != 224 || u16(52) != 64)
    return false; // EM_AMDGPU, ELF64
  auto range = [&](uint64_t offset, uint64_t size) {
    return offset <= n && size <= n - offset;
  };
  uint64_t ph = u64(32), sh = u64(40);
  auto pn = u16(56), sn = u16(60), ps = u16(54), ss = u16(58);
  if (!pn || pn == 65535 || !sn || !sh || ps != 56 || ss != 64 ||
      !range(ph, uint64_t(pn) * ps) || !range(sh, uint64_t(sn) * ss))
    return false;
  for (unsigned i = 0; i < pn; ++i) {
    auto at = size_t(ph + i * ps);
    if (!range(u64(at + 8), u64(at + 32)))
      return false;
  }
  for (unsigned i = 0; i < sn; ++i) {
    auto at = size_t(sh + i * ss);
    uint32_t type;
    std::memcpy(&type, p + at + 4, 4);
    if (type != 8 && !range(u64(at + 24), u64(at + 32)))
      return false;
  }
  return true;
}
} // namespace
extern "C" AmdResult amdModuleLoadData(AmdDevice d, const void *data, size_t n,
                                       AmdModule *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(d))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    if (!d || !data || !validElf(static_cast<const unsigned char *>(data), n))
      return AMD_ERROR_INVALID_VALUE;
    DeviceScope scope(d->index);
    if (scope.status)
      return result(scope.status);
    auto *v = amd_handles::track(new (std::nothrow) AmdModule_t{});
    if (!v)
      return AMD_ERROR_OUT_OF_MEMORY;
    v->device = d->index;
    auto r = result(hipModuleLoadData(&v->value, data));
    if (r)
      amd_handles::retire(v);
    else
      *out = v;
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdModuleLoadFile(AmdDevice d, const char *path,
                                       AmdModule *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(d))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    if (!d || !path)
      return AMD_ERROR_INVALID_VALUE;
    try {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file)
        return AMD_ERROR_MODULE_LOAD_FAILED;
      auto length = file.tellg();
      if (length <= 0 ||
          static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max())
        return AMD_ERROR_MODULE_LOAD_FAILED;
      std::vector<unsigned char> bytes(static_cast<size_t>(length));
      file.seekg(0);
      if (!file.read(reinterpret_cast<char *>(bytes.data()), length))
        return AMD_ERROR_MODULE_LOAD_FAILED;
      return amdModuleLoadData(d, bytes.data(), bytes.size(), out);
    } catch (const std::bad_alloc &) {
      return AMD_ERROR_OUT_OF_MEMORY;
    } catch (...) {
      return AMD_ERROR_MODULE_LOAD_FAILED;
    }

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdModuleUnload(AmdModule m) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(m))
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(m))
      return AMD_ERROR_INVALID_HANDLE;

    if (!m)
      return AMD_ERROR_INVALID_HANDLE;
    DeviceScope scope(m->device);
    if (scope.status)
      return result(scope.status);
    auto r = result(hipDeviceSynchronize());
    if (r)
      return r;
    r = result(hipModuleUnload(m->value));
    if (r)
      return r;
    while (m->functions) {
      auto f = m->functions;
      m->functions = f->next;
      amd_handles::retire(f);
    }
    amd_handles::retire(m);
    return AMD_SUCCESS;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdModuleGetFunction(AmdFunction *out, AmdModule m,
                                          const char *name) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(m))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    if (!m || !name)
      return AMD_ERROR_INVALID_VALUE;
    DeviceScope scope(m->device);
    if (scope.status)
      return result(scope.status);
    auto *v = amd_handles::track(new (std::nothrow) AmdFunction_t{});
    if (!v)
      return AMD_ERROR_OUT_OF_MEMORY;
    auto e = hipModuleGetFunction(&v->value, m->value, name);
    if (e != hipSuccess) {
      amd_handles::retire(v);
      return AMD_ERROR_SYMBOL_NOT_FOUND;
    }
    v->owner = m;
    v->next = m->functions;
    m->functions = v;
    *out = v;
    return AMD_SUCCESS;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdLaunchKernel(AmdFunction f, AmdDim3 g, AmdDim3 b,
                                     uint32_t sh, AmdStream s, void **args) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(f))
      return AMD_ERROR_INVALID_HANDLE;
    if (s && !amd_handles::valid(s))
      return AMD_ERROR_INVALID_HANDLE;

    if (!f)
      return AMD_ERROR_INVALID_HANDLE;
    if (!g.x || !g.y || !g.z || !b.x || !b.y || !b.z)
      return AMD_ERROR_INVALID_VALUE;
    if (s && s->device != f->owner->device)
      return AMD_ERROR_INVALID_VALUE;
    DeviceScope scope(f->owner->device);
    if (scope.status)
      return result(scope.status);
    return result(hipModuleLaunchKernel(f->value, g.x, g.y, g.z, b.x, b.y, b.z,
                                        sh, native(s), args, nullptr));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" const char *amdGetErrorString(AmdResult r) {
  switch (r) {
  case AMD_SUCCESS:
    return "success";
  case AMD_ERROR_INVALID_VALUE:
    return "invalid value";
  case AMD_ERROR_OUT_OF_MEMORY:
    return "out of memory";
  case AMD_ERROR_NOT_INITIALIZED:
    return "runtime not initialized";
  case AMD_ERROR_DEVICE_NOT_FOUND:
    return "device not found";
  case AMD_ERROR_INVALID_HANDLE:
    return "invalid handle";
  case AMD_ERROR_MODULE_LOAD_FAILED:
    return "module load failed";
  case AMD_ERROR_SYMBOL_NOT_FOUND:
    return "kernel symbol not found";
  case AMD_ERROR_LAUNCH_FAILED:
    return "kernel launch failed";
  case AMD_ERROR_NOT_READY:
    return "work pending or pool still has allocations";
  default:
    return "unknown AMD runtime error";
  }
}

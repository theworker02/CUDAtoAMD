#include "../runtime/compat_abi.h"
#include "amdcuda.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "handle_registry.h"
#include <vector>

#include <mutex>
#include <unordered_map>

#if defined(COMPATCUDA_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif

struct CompatStream {
#if defined(COMPATCUDA_ENABLE_HIP)
  hipStream_t native{};
#endif
};
struct CompatEvent {
#if defined(COMPATCUDA_ENABLE_HIP)
  hipEvent_t native{};
#endif
};
struct CompatGraph {
#if defined(COMPATCUDA_ENABLE_HIP)
  hipGraph_t native{};
#endif
};
struct AMDCUModule {
  std::vector<AMDCUfunction> functions;
#if defined(COMPATCUDA_ENABLE_HIP)
  hipModule_t native{};
#endif
};
struct AMDCUFunction {
#if defined(COMPATCUDA_ENABLE_HIP)
  hipFunction_t native{};
#endif
};

struct CompatContext {
#if defined(COMPATCUDA_ENABLE_HIP)
  hipCtx_t native{};
#endif
};
struct DriverFunction;
struct DriverModule {
#if defined(COMPATCUDA_ENABLE_HIP)
  hipModule_t native{};
#endif
  std::vector<DriverFunction *> functions;
};
struct DriverFunction {
#if defined(COMPATCUDA_ENABLE_HIP)
  hipFunction_t native{};
#endif
};
namespace {
thread_local cudaError_t last_error = cudaSuccess;
std::mutex allocation_mutex;
std::unordered_map<void *, std::size_t> allocations;

cudaError_t record(cudaError_t value) {
  if (value != cudaSuccess)
    last_error = value;
  return value;
}
cudaError_t unavailable() { return record(cudaErrorNotSupported); }

#if defined(COMPATCUDA_ENABLE_HIP)
cudaError_t from_hip(hipError_t error) {
  switch (error) {
  case hipSuccess:
    return record(cudaSuccess);
  case hipErrorInvalidValue:
    return record(cudaErrorInvalidValue);
  case hipErrorOutOfMemory:
    return record(cudaErrorMemoryAllocation);
  case hipErrorInvalidDevice:
    return record(cudaErrorInvalidDevice);
  case hipErrorNotReady:
    return record(cudaErrorNotReady);
  case hipErrorInvalidHandle:
    return record(cudaErrorInvalidResourceHandle);
  default:
    return record(cudaErrorUnknown);
  }
}
CUresult driver_from_hip(hipError_t e) {
  switch (e) {
  case hipSuccess:
    return CUDA_SUCCESS;
  case hipErrorInvalidValue:
    return CUDA_ERROR_INVALID_VALUE;
  case hipErrorOutOfMemory:
    return CUDA_ERROR_OUT_OF_MEMORY;
  case hipErrorInvalidDevice:
    return CUDA_ERROR_INVALID_DEVICE;
  case hipErrorInvalidContext:
    return CUDA_ERROR_INVALID_CONTEXT;
  case hipErrorInvalidHandle:
    return CUDA_ERROR_INVALID_HANDLE;
  case hipErrorNotReady:
    return CUDA_ERROR_NOT_READY;
  default:
    return CUDA_ERROR_UNKNOWN;
  }
}
bool copy_kind(cudaMemcpyKind value, hipMemcpyKind *output) {
  switch (value) {
  case cudaMemcpyHostToHost:
    *output = hipMemcpyHostToHost;
    return true;
  case cudaMemcpyHostToDevice:
    *output = hipMemcpyHostToDevice;
    return true;
  case cudaMemcpyDeviceToHost:
    *output = hipMemcpyDeviceToHost;
    return true;
  case cudaMemcpyDeviceToDevice:
    *output = hipMemcpyDeviceToDevice;
    return true;
  case cudaMemcpyDefault:
    *output = hipMemcpyDefault;
    return true;
  default:
    return false;
  }
}
#endif
} // namespace

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!count)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipGetDeviceCount(count));
#else
    *count = 0;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaSetDevice(int device) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipSetDevice(device));
#else
    (void)device;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaGetDevice(int *device) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!device)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipGetDevice(device));
#else
    *device = -1;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaDeviceSynchronize() {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipDeviceSynchronize());
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaGetLastError() {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    const auto value = last_error;
    last_error = cudaSuccess;
    return value;
  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" const char *cudaGetErrorString(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return "success";
  case cudaErrorNotSupported:
    return "compatibility operation is not supported by the active backend";
  case cudaErrorInvalidValue:
    return "invalid value";
  case cudaErrorMemoryAllocation:
    return "memory allocation failed";
  case cudaErrorInvalidDevice:
    return "invalid device";
  case cudaErrorNotReady:
    return "operation not ready";
  default:
    return "compatibility runtime error";
  }
}
extern "C" cudaError_t cudaMalloc(void **pointer, std::size_t bytes) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!pointer)
      return record(cudaErrorInvalidValue);
    *pointer = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    const auto result = from_hip(hipMalloc(pointer, bytes));
    if (result == cudaSuccess && *pointer) {
      try {
        std::lock_guard lock(allocation_mutex);
        allocations.emplace(*pointer, bytes);
      } catch (...) {
        hipFree(*pointer);
        *pointer = nullptr;
        throw;
      }
    }
    return result;
#else
    (void)bytes;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaFree(void *pointer) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!pointer)
      return record(cudaSuccess);
#if defined(COMPATCUDA_ENABLE_HIP)
    const auto result = from_hip(hipFree(pointer));
    if (result == cudaSuccess) {
      std::lock_guard lock(allocation_mutex);
      allocations.erase(pointer);
    }
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaMemcpy(void *destination, const void *source,
                                  std::size_t bytes, cudaMemcpyKind kind) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if ((!destination || !source) && bytes != 0)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    hipMemcpyKind native_kind{};
    if (!copy_kind(kind, &native_kind))
      return record(cudaErrorInvalidValue);
    return from_hip(hipMemcpy(destination, source, bytes, native_kind));
#else
    (void)destination;
    (void)source;
    (void)bytes;
    (void)kind;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaMemcpyAsync(void *destination, const void *source,
                                       std::size_t bytes, cudaMemcpyKind kind,
                                       cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);

    if ((!destination || !source) && bytes != 0)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    hipMemcpyKind native_kind{};
    if (!copy_kind(kind, &native_kind))
      return record(cudaErrorInvalidValue);
    return from_hip(hipMemcpyAsync(destination, source, bytes, native_kind,
                                   stream ? stream->native : nullptr));
#else
    (void)destination;
    (void)source;
    (void)bytes;
    (void)kind;
    (void)stream;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaMemset(void *pointer, int value, std::size_t bytes) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!pointer && bytes != 0)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipMemset(pointer, value, bytes));
#else
    (void)pointer;
    (void)value;
    (void)bytes;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaMemsetAsync(void *pointer, int value,
                                       std::size_t bytes, cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);

    if (!pointer && bytes != 0)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipMemsetAsync(pointer, value, bytes,
                                   stream ? stream->native : nullptr));
#else
    (void)pointer;
    (void)value;
    (void)bytes;
    (void)stream;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!stream)
      return record(cudaErrorInvalidValue);
    *stream = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = amd_handles::track(new (std::nothrow) CompatStream{});
    if (!value)
      return record(cudaErrorMemoryAllocation);
    const auto result = from_hip(hipStreamCreate(&value->native));
    if (result != cudaSuccess) {
      amd_handles::retire(value);
      return result;
    }
    *stream = value;
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);
    if (amd_handles::pinned(stream))
      return record(cudaErrorNotReady);

    if (!stream)
      return record(cudaSuccess);
#if defined(COMPATCUDA_ENABLE_HIP)
    const auto result = from_hip(hipStreamDestroy(stream->native));
    if (result == cudaSuccess)
      amd_handles::retire(stream);
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);

#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipStreamSynchronize(stream ? stream->native : nullptr));
#else
    (void)stream;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaEventCreate(cudaEvent_t *event) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!event)
      return record(cudaErrorInvalidValue);
    *event = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = amd_handles::track(new (std::nothrow) CompatEvent{});
    if (!value)
      return record(cudaErrorMemoryAllocation);
    const auto result = from_hip(hipEventCreate(&value->native));
    if (result != cudaSuccess) {
      amd_handles::retire(value);
      return result;
    }
    *event = value;
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (event && !amd_handles::valid(event))
      return record(cudaErrorInvalidResourceHandle);
    if (amd_handles::pinned(event))
      return record(cudaErrorNotReady);

    if (!event)
      return record(cudaSuccess);
#if defined(COMPATCUDA_ENABLE_HIP)
    const auto result = from_hip(hipEventDestroy(event->native));
    if (result == cudaSuccess)
      amd_handles::retire(event);
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (event && !amd_handles::valid(event))
      return record(cudaErrorInvalidResourceHandle);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);

    if (!event)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(
        hipEventRecord(event->native, stream ? stream->native : nullptr));
#else
    (void)stream;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (event && !amd_handles::valid(event))
      return record(cudaErrorInvalidResourceHandle);

    if (!event)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipEventSynchronize(event->native));
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaGraphCreate(cudaGraph_t *graph, unsigned int flags) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!graph)
      return record(cudaErrorInvalidValue);
    *graph = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = amd_handles::track(new (std::nothrow) CompatGraph{});
    if (!value)
      return record(cudaErrorMemoryAllocation);
    const auto result = from_hip(hipGraphCreate(&value->native, flags));
    if (result != cudaSuccess) {
      amd_handles::retire(value);
      return result;
    }
    *graph = value;
    return result;
#else
    (void)flags;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (graph && !amd_handles::valid(graph))
      return record(cudaErrorInvalidResourceHandle);
    if (amd_handles::pinned(graph))
      return record(cudaErrorNotReady);

    if (!graph)
      return record(cudaSuccess);
#if defined(COMPATCUDA_ENABLE_HIP)
    const auto result = from_hip(hipGraphDestroy(graph->native));
    if (result == cudaSuccess)
      amd_handles::retire(graph);
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" CUresult cuInit(unsigned int flags) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

#if defined(COMPATCUDA_ENABLE_HIP)
    return driver_from_hip(hipInit(flags));
#else
    (void)flags;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuDeviceGetCount(int *count) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!count)
      return CUDA_ERROR_INVALID_VALUE;
#if defined(COMPATCUDA_ENABLE_HIP)
    return driver_from_hip(hipGetDeviceCount(count));
#else
    *count = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!device)
      return CUDA_ERROR_INVALID_VALUE;
#if defined(COMPATCUDA_ENABLE_HIP)
    return driver_from_hip(hipDeviceGet(device, ordinal));
#else
    (void)ordinal;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuCtxCreate(CUcontext *context, unsigned int flags,
                                CUdevice device) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!context)
      return CUDA_ERROR_INVALID_VALUE;
    *context = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = amd_handles::track(new (std::nothrow) CompatContext{});
    if (!value)
      return CUDA_ERROR_OUT_OF_MEMORY;
    const auto result =
        driver_from_hip(hipCtxCreate(&value->native, flags, device));
    if (result == CUDA_SUCCESS)
      *context = value;
    else
      amd_handles::retire(value);
    return static_cast<CUresult>(result);
#else
    (void)flags;
    (void)device;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuCtxDestroy(CUcontext context) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = static_cast<CompatContext *>(context);
    if (!amd_handles::valid(value))
      return CUDA_ERROR_INVALID_CONTEXT;
    auto result = driver_from_hip(hipCtxDestroy(value->native));
    if (!result)
      amd_handles::retire(value);
    return result;
#else
    (void)context;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuMemAlloc(CUdeviceptr *pointer, std::size_t bytes) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!pointer)
      return CUDA_ERROR_INVALID_VALUE;
    *pointer = 0;
#if defined(COMPATCUDA_ENABLE_HIP)
    void *native{};
    const auto result = driver_from_hip(hipMalloc(&native, bytes));
    if (result == cudaSuccess)
      *pointer = reinterpret_cast<CUdeviceptr>(native);
    return static_cast<CUresult>(result);
#else
    (void)bytes;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuMemFree(CUdeviceptr pointer) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

#if defined(COMPATCUDA_ENABLE_HIP)
    return driver_from_hip(hipFree(reinterpret_cast<void *>(pointer)));
#else
    (void)pointer;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuMemcpyHtoD(CUdeviceptr destination, const void *source,
                                 std::size_t bytes) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!source && bytes != 0)
      return CUDA_ERROR_INVALID_VALUE;
#if defined(COMPATCUDA_ENABLE_HIP)
    return driver_from_hip(hipMemcpy(reinterpret_cast<void *>(destination),
                                     source, bytes, hipMemcpyHostToDevice));
#else
    (void)destination;
    (void)source;
    (void)bytes;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuMemcpyDtoH(void *destination, CUdeviceptr source,
                                 std::size_t bytes) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!destination && bytes != 0)
      return CUDA_ERROR_INVALID_VALUE;
#if defined(COMPATCUDA_ENABLE_HIP)
    return driver_from_hip(hipMemcpy(destination,
                                     reinterpret_cast<void *>(source), bytes,
                                     hipMemcpyDeviceToHost));
#else
    (void)destination;
    (void)source;
    (void)bytes;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuModuleLoad(CUmodule *module, const char *path) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!module || !path)
      return CUDA_ERROR_INVALID_VALUE;
    *module = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = amd_handles::track(new (std::nothrow) DriverModule{});
    if (!value)
      return CUDA_ERROR_OUT_OF_MEMORY;
    const auto result = driver_from_hip(hipModuleLoad(&value->native, path));
    if (result == CUDA_SUCCESS)
      *module = value;
    else
      amd_handles::retire(value);
    return static_cast<CUresult>(result);
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuModuleUnload(CUmodule module) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = static_cast<DriverModule *>(module);
    if (!amd_handles::valid(value))
      return CUDA_ERROR_INVALID_HANDLE;
    auto result = driver_from_hip(hipDeviceSynchronize());
    if (result)
      return result;
    result = driver_from_hip(hipModuleUnload(value->native));
    if (!result) {
      for (auto f : value->functions)
        amd_handles::retire(f);
      std::vector<DriverFunction *>().swap(value->functions);
      amd_handles::retire(value);
    }
    return result;
#else
    (void)module;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" cudaError_t amdcuModuleLoad(AMDCUmodule *module, const char *path) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);

    if (!module || !path)
      return record(cudaErrorInvalidValue);
    *module = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = amd_handles::track(new (std::nothrow) AMDCUModule{});
    if (!value)
      return record(cudaErrorMemoryAllocation);
    const auto result = from_hip(hipModuleLoad(&value->native, path));
    if (result != cudaSuccess) {
      amd_handles::retire(value);
      return result;
    }
    *module = value;
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t amdcuModuleUnload(AMDCUmodule module) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (module && !amd_handles::valid(module))
      return record(cudaErrorInvalidResourceHandle);
    if (amd_handles::pinned(module))
      return record(cudaErrorNotReady);

    if (!module)
      return record(cudaSuccess);
#if defined(COMPATCUDA_ENABLE_HIP)
    auto result = from_hip(hipDeviceSynchronize());
    if (result != cudaSuccess)
      return result;
    result = from_hip(hipModuleUnload(module->native));
    if (result == cudaSuccess) {
      for (auto f : module->functions)
        amd_handles::retire(f);
      std::vector<AMDCUfunction>().swap(module->functions);
      amd_handles::retire(module);
    }
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t amdcuModuleGetFunction(AMDCUfunction *function,
                                              AMDCUmodule module,
                                              const char *name) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (module && !amd_handles::valid(module))
      return record(cudaErrorInvalidResourceHandle);

    if (!function || !module || !name)
      return record(cudaErrorInvalidValue);
    *function = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = amd_handles::track(new (std::nothrow) AMDCUFunction{});
    if (!value)
      return record(cudaErrorMemoryAllocation);
    const auto result =
        from_hip(hipModuleGetFunction(&value->native, module->native, name));
    if (result != cudaSuccess) {
      amd_handles::retire(value);
      return result;
    }
    try {
      module->functions.push_back(value);
    } catch (...) {
      amd_handles::retire(value);
      throw;
    }
    *function = value;
    return result;
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t
amdcuLaunchKernel(AMDCUfunction function, unsigned int gx, unsigned int gy,
                  unsigned int gz, unsigned int bx, unsigned int by,
                  unsigned int bz, unsigned int shared, cudaStream_t stream,
                  void **arguments) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (function && !amd_handles::valid(function))
      return record(cudaErrorInvalidResourceHandle);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);

    if (!function)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipModuleLaunchKernel(
        function->native, gx, gy, gz, bx, by, bz, shared,
        stream ? stream->native : nullptr, arguments, nullptr));
#else
    (void)gx;
    (void)gy;
    (void)gz;
    (void)bx;
    (void)by;
    (void)bz;
    (void)shared;
    (void)stream;
    (void)arguments;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t amdcuMallocAsync(void **pointer, std::size_t bytes,
                                        cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);

    if (!pointer)
      return record(cudaErrorInvalidValue);
    *pointer = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(
        hipMallocAsync(pointer, bytes, stream ? stream->native : nullptr));
#else
    (void)bytes;
    (void)stream;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t amdcuFreeAsync(void *pointer, cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);
    if (amd_handles::pinned(stream))
      return record(cudaErrorNotReady);

#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipFreeAsync(pointer, stream ? stream->native : nullptr));
#else
    (void)pointer;
    (void)stream;
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaPeekAtLastError() {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    return last_error;
  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);

#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipStreamQuery(stream ? stream->native : nullptr));
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                           cudaEvent_t event,
                                           unsigned int flags) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (stream && !amd_handles::valid(stream))
      return record(cudaErrorInvalidResourceHandle);
    if (event && !amd_handles::valid(event))
      return record(cudaErrorInvalidResourceHandle);

    if (!event || flags)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipStreamWaitEvent(stream ? stream->native : nullptr,
                                       event->native, flags));
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (event && !amd_handles::valid(event))
      return record(cudaErrorInvalidResourceHandle);

    if (!event)
      return record(cudaErrorInvalidResourceHandle);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipEventQuery(event->native));
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                            cudaEvent_t stop) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return record(cudaErrorNotReady);
    if (start && !amd_handles::valid(start))
      return record(cudaErrorInvalidResourceHandle);
    if (stop && !amd_handles::valid(stop))
      return record(cudaErrorInvalidResourceHandle);

    if (!ms || !start || !stop)
      return record(cudaErrorInvalidValue);
#if defined(COMPATCUDA_ENABLE_HIP)
    return from_hip(hipEventElapsedTime(ms, start->native, stop->native));
#else
    return unavailable();
#endif

  } catch (const std::bad_alloc &) {
    return record(cudaErrorMemoryAllocation);
  } catch (...) {
    return record(cudaErrorUnknown);
  }
}
extern "C" CUresult cuModuleGetFunction(CUfunction *out, CUmodule module,
                                        const char *name) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;

    if (!out)
      return CUDA_ERROR_INVALID_VALUE;
    *out = nullptr;
#if defined(COMPATCUDA_ENABLE_HIP)
    auto *owner = static_cast<DriverModule *>(module);
    if (!amd_handles::valid(owner))
      return CUDA_ERROR_INVALID_HANDLE;
    if (!name)
      return CUDA_ERROR_INVALID_VALUE;
    auto *value = amd_handles::track(new (std::nothrow) DriverFunction{});
    if (!value)
      return CUDA_ERROR_OUT_OF_MEMORY;
    auto e = hipModuleGetFunction(&value->native, owner->native, name);
    if (e != hipSuccess) {
      amd_handles::retire(value);
      return CUDA_ERROR_NOT_FOUND;
    }
    try {
      owner->functions.push_back(value);
    } catch (...) {
      amd_handles::retire(value);
      throw;
    }
    *out = value;
    return CUDA_SUCCESS;
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult cuLaunchKernel(CUfunction function, unsigned int gx,
                                   unsigned int gy, unsigned int gz,
                                   unsigned int bx, unsigned int by,
                                   unsigned int bz, unsigned int shared,
                                   cudaStream_t stream, void **arguments,
                                   void **extra) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return CUDA_ERROR_NOT_READY;
    if (stream && !amd_handles::valid(stream))
      return CUDA_ERROR_INVALID_HANDLE;

#if defined(COMPATCUDA_ENABLE_HIP)
    auto *value = static_cast<DriverFunction *>(function);
    if (!amd_handles::valid(value))
      return CUDA_ERROR_INVALID_HANDLE;
    if (!gx || !gy || !gz || !bx || !by || !bz)
      return CUDA_ERROR_INVALID_VALUE;
    if (extra)
      return CUDA_ERROR_NOT_SUPPORTED;
    return driver_from_hip(hipModuleLaunchKernel(
        value->native, gx, gy, gz, bx, by, bz, shared,
        stream ? stream->native : nullptr, arguments, nullptr));
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif

  } catch (const std::bad_alloc &) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}

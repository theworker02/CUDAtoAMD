#include "amd_fft.h"
#include "handle_registry.h"
#ifdef COMPATCUDA_ENABLE_HIPFFT
#include "hip_interop.h"
#include <hipfft/hipfft.h>
#include <limits>
#include <new>
struct AmdFftPlan_t {
  hipfftHandle value;
  int device;
  uint64_t bytes;
};
namespace {
AmdResult fft_result(hipfftResult value) {
  switch (value) {
  case HIPFFT_SUCCESS:
    return AMD_SUCCESS;
  case HIPFFT_ALLOC_FAILED:
    return AMD_ERROR_OUT_OF_MEMORY;
  case HIPFFT_INVALID_VALUE:
  case HIPFFT_INVALID_SIZE:
    return AMD_ERROR_INVALID_VALUE;
  default:
    return AMD_ERROR_UNKNOWN;
  }
}
} // namespace
AmdResult amdFftPlan1d(AmdDevice device, int length, int batches,
                       AmdFftPlan *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(device))
      return AMD_ERROR_INVALID_HANDLE;

    if (!out)
      return AMD_ERROR_INVALID_VALUE;
    *out = nullptr;
    int index;
    if (length < 1 || batches < 1 || !amd_native::device_index(device, &index))
      return AMD_ERROR_INVALID_VALUE;
    auto elements = uint64_t(length) * uint64_t(batches);
    if (elements > std::numeric_limits<size_t>::max() / sizeof(AmdComplex32))
      return AMD_ERROR_INVALID_VALUE;
    amd_native::DeviceGuard guard(index);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    auto *plan = amd_handles::track(new (std::nothrow) AmdFftPlan_t{});
    if (!plan)
      return AMD_ERROR_OUT_OF_MEMORY;
    plan->device = index;
    plan->bytes = elements * sizeof(AmdComplex32);
    auto r =
        fft_result(hipfftPlan1d(&plan->value, length, HIPFFT_C2C, batches));
    if (r)
      amd_handles::retire(plan);
    else
      *out = plan;
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdFftExecC2C(AmdFftPlan plan, AmdStream stream, AmdDeviceAddr input,
                        AmdDeviceAddr output, int direction) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(plan))
      return AMD_ERROR_INVALID_HANDLE;
    if (stream && !amd_handles::valid(stream))
      return AMD_ERROR_INVALID_HANDLE;

    if (!plan)
      return AMD_ERROR_INVALID_HANDLE;
    hipStream_t native{};
    if (!input || !output || (direction != -1 && direction != 1) ||
        !amd_native::stream_handle(stream, plan->device, &native))
      return AMD_ERROR_INVALID_VALUE;
    if (input > UINT64_MAX - plan->bytes || output > UINT64_MAX - plan->bytes ||
        (input != output && input < output + plan->bytes &&
         output < input + plan->bytes))
      return AMD_ERROR_INVALID_VALUE;
    amd_native::DeviceGuard guard(plan->device);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    auto r = fft_result(hipfftSetStream(plan->value, native));
    if (r)
      return r;
    return fft_result(
        hipfftExecC2C(plan->value, reinterpret_cast<hipfftComplex *>(input),
                      reinterpret_cast<hipfftComplex *>(output), direction));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdFftDestroy(AmdFftPlan plan) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(plan))
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(plan))
      return AMD_ERROR_INVALID_HANDLE;

    if (!plan)
      return AMD_ERROR_INVALID_HANDLE;
    amd_native::DeviceGuard guard(plan->device);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    if (hipDeviceSynchronize() != hipSuccess)
      return AMD_ERROR_UNKNOWN;
    auto r = fft_result(hipfftDestroy(plan->value));
    if (!r)
      amd_handles::retire(plan);
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
#else
AmdResult amdFftPlan1d(AmdDevice, int, int, AmdFftPlan *out) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

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
AmdResult amdFftExecC2C(AmdFftPlan, AmdStream, AmdDeviceAddr, AmdDeviceAddr,
                        int) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    return AMD_ERROR_NOT_INITIALIZED;
  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
AmdResult amdFftDestroy(AmdFftPlan) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;

    return AMD_ERROR_NOT_INITIALIZED;
  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
#endif

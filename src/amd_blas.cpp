#include "amd_blas.h"
#include "handle_registry.h"
#include "hip_interop.h"
#include <algorithm>
#include <cstdint>
#include <hipblas/hipblas.h>
#include <limits>
#include <new>
struct AmdBlasHandle_t {
  hipblasHandle_t value;
  int device;
};
namespace {
AmdResult result(hipblasStatus_t s) {
  switch (s) {
  case HIPBLAS_STATUS_SUCCESS:
    return AMD_SUCCESS;
  case HIPBLAS_STATUS_INVALID_VALUE:
    return AMD_ERROR_INVALID_VALUE;
  case HIPBLAS_STATUS_ALLOC_FAILED:
    return AMD_ERROR_OUT_OF_MEMORY;
  default:
    return AMD_ERROR_UNKNOWN;
  }
}
hipblasOperation_t op(int value) {
  return value == 1 ? HIPBLAS_OP_T : HIPBLAS_OP_N;
}
bool valid(int ta, int tb, int m, int n, int k, int lda, int ldb, int ldc) {
  return (ta == 0 || ta == 1) && (tb == 0 || tb == 1) && m >= 0 && n >= 0 &&
         k >= 0 && lda >= std::max(1, ta ? k : m) &&
         ldb >= std::max(1, tb ? n : k) && ldc >= std::max(1, m);
}
bool stride_ok(long long stride, long long span, int batches, int bytes,
               bool output) {
  if (stride < 0 || (batches > 1 && (output || stride) && stride < span))
    return false;
  const auto max = std::numeric_limits<long long>::max() / bytes;
  return span <= max &&
         (batches <= 1 || stride <= (max - span) / (batches - 1));
}
} // namespace
extern "C" AmdResult amdBlasCreate(AmdBlasHandle *out) {
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
    auto *v = amd_handles::track(new (std::nothrow) AmdBlasHandle_t{});
    if (!v)
      return AMD_ERROR_OUT_OF_MEMORY;
    if (hipGetDevice(&v->device) != hipSuccess) {
      amd_handles::retire(v);
      return AMD_ERROR_NOT_INITIALIZED;
    }
    auto r = result(hipblasCreate(&v->value));
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
extern "C" AmdResult amdBlasCreateOnDevice(AmdDevice device,
                                           AmdBlasHandle *out) {
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
    int index;
    if (!amd_native::device_index(device, &index))
      return AMD_ERROR_INVALID_HANDLE;
    amd_native::DeviceGuard guard(index);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    return amdBlasCreate(out);

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdBlasDestroy(AmdBlasHandle h) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;
    if (amd_handles::capturing)
      return AMD_ERROR_NOT_READY;
    if (amd_handles::pinned(h))
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(h))
      return AMD_ERROR_INVALID_HANDLE;

    if (!h)
      return AMD_ERROR_INVALID_HANDLE;
    amd_native::DeviceGuard guard(h->device);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    if (hipDeviceSynchronize() != hipSuccess)
      return AMD_ERROR_UNKNOWN;
    auto r = result(hipblasDestroy(h->value));
    if (!r)
      amd_handles::retire(h);
    return r;

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdBlasGemmStridedBatchedEx(
    AmdBlasHandle h, AmdStream stream, AmdBlasDataType type, int ta, int tb,
    int m, int n, int k, float alpha, AmdDeviceAddr a, int lda, long long sa,
    AmdDeviceAddr b, int ldb, long long sb, float beta, AmdDeviceAddr c,
    int ldc, long long sc, int batches) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(h))
      return AMD_ERROR_INVALID_HANDLE;
    if (stream && !amd_handles::valid(stream))
      return AMD_ERROR_INVALID_HANDLE;

    if (!h)
      return AMD_ERROR_INVALID_HANDLE;
    if (!valid(ta, tb, m, n, k, lda, ldb, ldc) || batches < 0 ||
        type < AMD_BLAS_F32 || type > AMD_BLAS_BF16)
      return AMD_ERROR_INVALID_VALUE;
    int bytes = type == AMD_BLAS_F32 ? 4 : 2;
    if (!stride_ok(sa, (long long)lda * (ta ? m : k), batches, bytes, false) ||
        !stride_ok(sb, (long long)ldb * (tb ? k : n), batches, bytes, false) ||
        !stride_ok(sc, (long long)ldc * n, batches, 4, true))
      return AMD_ERROR_INVALID_VALUE;
    hipStream_t native{};
    if (!amd_native::stream_handle(stream, h->device, &native))
      return AMD_ERROR_INVALID_VALUE;
    if (!m || !n || !batches)
      return AMD_SUCCESS;
    if (!c || (k && (!a || !b)))
      return AMD_ERROR_INVALID_VALUE;
    amd_native::DeviceGuard guard(h->device);
    if (guard.status)
      return AMD_ERROR_DEVICE_NOT_FOUND;
    auto r = result(hipblasSetStream(h->value, native));
    if (r)
      return r;
    auto dtype = type == AMD_BLAS_F32   ? HIP_R_32F
                 : type == AMD_BLAS_F16 ? HIP_R_16F
                                        : HIP_R_16BF;
    return result(hipblasGemmStridedBatchedEx(
        h->value, op(ta), op(tb), m, n, k, &alpha, (const void *)a, dtype, lda,
        sa, (const void *)b, dtype, ldb, sb, &beta, (void *)c, HIP_R_32F, ldc,
        sc, batches, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdBlasSgemm(AmdBlasHandle h, AmdBlasOperation ta,
                                  AmdBlasOperation tb, int m, int n, int k,
                                  float alpha, AmdDeviceAddr a, int lda,
                                  AmdDeviceAddr b, int ldb, float beta,
                                  AmdDeviceAddr c, int ldc) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(h))
      return AMD_ERROR_INVALID_HANDLE;

    return amdBlasGemmStridedBatchedEx(h, nullptr, AMD_BLAS_F32, ta, tb, m, n,
                                       k, alpha, a, lda, 0, b, ldb, 0, beta, c,
                                       ldc, 0, 1);

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}
extern "C" AmdResult amdBlasSgemmStridedBatched(
    AmdBlasHandle h, AmdBlasOperation ta, AmdBlasOperation tb, int m, int n,
    int k, float alpha, AmdDeviceAddr a, int lda, long long sa, AmdDeviceAddr b,
    int ldb, long long sb, float beta, AmdDeviceAddr c, int ldc, long long sc,
    int batches) {
  try {
    std::lock_guard<std::recursive_mutex> guard_lock(amd_handles::mutex);
    if (amd_handles::capturing &&
        amd_handles::capture_thread != std::this_thread::get_id())
      return AMD_ERROR_NOT_READY;

    if (!amd_handles::valid(h))
      return AMD_ERROR_INVALID_HANDLE;

    return amdBlasGemmStridedBatchedEx(h, nullptr, AMD_BLAS_F32, ta, tb, m, n,
                                       k, alpha, a, lda, sa, b, ldb, sb, beta,
                                       c, ldc, sc, batches);

  } catch (const std::bad_alloc &) {
    return AMD_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return AMD_ERROR_UNKNOWN;
  }
}

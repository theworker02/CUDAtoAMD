#include "amd_fft.h"
#include "cublas_v2.h"
#include "cufft.h"
#include "handle_registry.h"
#include <climits>
#include <unordered_map>
namespace {
cublasStatus_t blas_status(AmdResult r) {
  switch (r) {
  case AMD_SUCCESS:
    return CUBLAS_STATUS_SUCCESS;
  case AMD_ERROR_INVALID_HANDLE:
  case AMD_ERROR_NOT_INITIALIZED:
    return CUBLAS_STATUS_NOT_INITIALIZED;
  case AMD_ERROR_OUT_OF_MEMORY:
    return CUBLAS_STATUS_ALLOC_FAILED;
  case AMD_ERROR_INVALID_VALUE:
    return CUBLAS_STATUS_INVALID_VALUE;
  default:
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
}
cufftResult fft_status(AmdResult r) {
  switch (r) {
  case AMD_SUCCESS:
    return CUFFT_SUCCESS;
  case AMD_ERROR_INVALID_HANDLE:
    return CUFFT_INVALID_PLAN;
  case AMD_ERROR_NOT_INITIALIZED:
    return CUFFT_NOT_SUPPORTED;
  case AMD_ERROR_OUT_OF_MEMORY:
    return CUFFT_ALLOC_FAILED;
  case AMD_ERROR_INVALID_VALUE:
    return CUFFT_INVALID_VALUE;
  default:
    return CUFFT_INTERNAL_ERROR;
  }
}
std::unordered_map<int, AmdFftPlan> plans;
int next_plan = 1;
} // namespace
cublasStatus_t cublasCreate(cublasHandle_t *h) {
  return blas_status(amdBlasCreate(h));
}
cublasStatus_t cublasDestroy(cublasHandle_t h) {
  return blas_status(amdBlasDestroy(h));
}
cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta,
                           cublasOperation_t tb, int m, int n, int k,
                           const float *alpha, const float *a, int lda,
                           const float *b, int ldb, const float *beta, float *c,
                           int ldc) {
  if (!alpha || !beta || ta < 0 || ta > 2 || tb < 0 || tb > 2)
    return CUBLAS_STATUS_INVALID_VALUE;
  return blas_status(
      amdBlasSgemm(h, ta == CUBLAS_OP_N ? AMD_BLAS_OP_N : AMD_BLAS_OP_T,
                   tb == CUBLAS_OP_N ? AMD_BLAS_OP_N : AMD_BLAS_OP_T, m, n, k,
                   *alpha, (AmdDeviceAddr)a, lda, (AmdDeviceAddr)b, ldb, *beta,
                   (AmdDeviceAddr)c, ldc));
}
cufftResult cufftPlan1d(cufftHandle *out, int length, cufftType type,
                        int batch) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    if (!out)
      return CUFFT_INVALID_VALUE;
    *out = 0;
    if (type != CUFFT_C2C)
      return CUFFT_INVALID_TYPE;
    if (next_plan == INT_MAX)
      return CUFFT_ALLOC_FAILED;
    int index = 0;
    if (cudaGetDevice(&index) != cudaSuccess)
      return CUFFT_NOT_SUPPORTED;
    AmdDevice device{};
    auto r = amdGetDevice(&device, index);
    if (r)
      return fft_status(r);
    AmdFftPlan plan{};
    r = amdFftPlan1d(device, length, batch, &plan);
    amdDeviceDestroy(device);
    if (r)
      return fft_status(r);
    try {
      plans.emplace(next_plan, plan);
    } catch (...) {
      amdFftDestroy(plan);
      throw;
    }
    *out = next_plan++;
    return CUFFT_SUCCESS;
  } catch (...) {
    return CUFFT_ALLOC_FAILED;
  }
}
cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
    int k, const void *alpha, const void *a, cudaDataType at, int lda,
    long long sa, const void *b, cudaDataType bt, int ldb, long long sb,
    const void *beta, void *c, cudaDataType ct, int ldc, long long sc,
    int batches, cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
  if (!alpha || !beta || ta < 0 || ta > 2 || tb < 0 || tb > 2)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (at != bt || ct != CUDA_R_32F || compute != CUBLAS_COMPUTE_32F ||
      algorithm != CUBLAS_GEMM_DEFAULT)
    return CUBLAS_STATUS_NOT_SUPPORTED;
  AmdBlasDataType type;
  switch (at) {
  case CUDA_R_32F:
    type = AMD_BLAS_F32;
    break;
  case CUDA_R_16F:
    type = AMD_BLAS_F16;
    break;
  case CUDA_R_16BF:
    type = AMD_BLAS_BF16;
    break;
  default:
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return blas_status(amdBlasGemmStridedBatchedEx(
      h, nullptr, type, ta != CUBLAS_OP_N, tb != CUBLAS_OP_N, m, n, k,
      *static_cast<const float *>(alpha), reinterpret_cast<AmdDeviceAddr>(a),
      lda, sa, reinterpret_cast<AmdDeviceAddr>(b), ldb, sb,
      *static_cast<const float *>(beta), reinterpret_cast<AmdDeviceAddr>(c),
      ldc, sc, batches));
}
cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta,
                            cublasOperation_t tb, int m, int n, int k,
                            const void *alpha, const void *a, cudaDataType at,
                            int lda, const void *b, cudaDataType bt, int ldb,
                            const void *beta, void *c, cudaDataType ct, int ldc,
                            cublasComputeType_t compute,
                            cublasGemmAlgo_t algorithm) {
  return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, a, at, lda, 0, b,
                                    bt, ldb, 0, beta, c, ct, ldc, 0, 1, compute,
                                    algorithm);
}
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta,
                                         cublasOperation_t tb, int m, int n,
                                         int k, const float *alpha,
                                         const float *a, int lda, long long sa,
                                         const float *b, int ldb, long long sb,
                                         const float *beta, float *c, int ldc,
                                         long long sc, int batches) {
  return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, a, CUDA_R_32F,
                                    lda, sa, b, CUDA_R_32F, ldb, sb, beta, c,
                                    CUDA_R_32F, ldc, sc, batches,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
cufftResult cufftExecC2C(cufftHandle id, cufftComplex *input,
                         cufftComplex *output, int direction) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    auto it = plans.find(id);
    if (it == plans.end())
      return CUFFT_INVALID_PLAN;
    return fft_status(amdFftExecC2C(it->second, nullptr, (AmdDeviceAddr)input,
                                    (AmdDeviceAddr)output, direction));
  } catch (...) {
    return CUFFT_INTERNAL_ERROR;
  }
}
cufftResult cufftDestroy(cufftHandle id) {
  try {
    std::lock_guard<std::recursive_mutex> lock(amd_handles::mutex);
    auto it = plans.find(id);
    if (it == plans.end())
      return CUFFT_INVALID_PLAN;
    auto r = amdFftDestroy(it->second);
    if (!r)
      plans.erase(it);
    return fft_status(r);
  } catch (...) {
    return CUFFT_INTERNAL_ERROR;
  }
}

#pragma once
#include "amd_blas.h"
#include "cuda_runtime_api.h"
#include "library_types.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef AmdBlasHandle cublasHandle_t;
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15
} cublasStatus_t;
typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2
} cublasOperation_t;
typedef enum { CUBLAS_COMPUTE_32F = 68 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT = -1 } cublasGemmAlgo_t;
/* Supported Ex combinations: equal F32/F16/BF16 A/B, F32 C/scalars/compute.
   Default stream and algorithm only. Strides are in storage elements. */
COMPATCUDA_API cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void *, const void *, cudaDataType, int, long long, const void *,
    cudaDataType, int, long long, const void *, void *, cudaDataType, int,
    long long, int, cublasComputeType_t, cublasGemmAlgo_t);
COMPATCUDA_API cublasStatus_t
cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int,
             int, const void *, const void *, cudaDataType, int, const void *,
             cudaDataType, int, const void *, void *, cudaDataType, int,
             cublasComputeType_t, cublasGemmAlgo_t);
COMPATCUDA_API cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const float *, const float *, int, long long, const float *, int, long long,
    const float *, float *, int, long long, int);
COMPATCUDA_API cublasStatus_t cublasCreate(cublasHandle_t *handle);
COMPATCUDA_API cublasStatus_t cublasDestroy(cublasHandle_t handle);
/* Host scalar pointer mode; default stream only in this subset. */
COMPATCUDA_API cublasStatus_t
cublasSgemm(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb,
            int m, int n, int k, const float *alpha, const float *a, int lda,
            const float *b, int ldb, const float *beta, float *c, int ldc);
#ifdef __cplusplus
}
#endif

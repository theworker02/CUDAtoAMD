#pragma once
#include "amd_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AmdBlasHandle_t* AmdBlasHandle;
/* Ex GEMM uses FP32 scalars, accumulation and output, with this A/B storage type. */
typedef enum AmdBlasDataType { AMD_BLAS_F32=0, AMD_BLAS_F16=1, AMD_BLAS_BF16=2 } AmdBlasDataType;
AmdResult amdBlasCreateOnDevice(AmdDevice device, AmdBlasHandle* handle);
AmdResult amdBlasGemmStridedBatchedEx(AmdBlasHandle handle,AmdStream stream,AmdBlasDataType input_type,int trans_a,int trans_b,int m,int n,int k,float alpha,AmdDeviceAddr a,int lda,long long stride_a,AmdDeviceAddr b,int ldb,long long stride_b,float beta,AmdDeviceAddr c,int ldc,long long stride_c,int batch_count);
typedef enum AmdBlasOperation { AMD_BLAS_OP_N = 0, AMD_BLAS_OP_T = 1 } AmdBlasOperation;
AmdResult amdBlasCreate(AmdBlasHandle* handle); AmdResult amdBlasDestroy(AmdBlasHandle handle);
AmdResult amdBlasSgemm(AmdBlasHandle handle,AmdBlasOperation trans_a,AmdBlasOperation trans_b,int m,int n,int k,float alpha,AmdDeviceAddr a,int lda,AmdDeviceAddr b,int ldb,float beta,AmdDeviceAddr c,int ldc);
/* Column-major, constant-stride batches. Strides are measured in float elements. */
AmdResult amdBlasSgemmStridedBatched(AmdBlasHandle handle,AmdBlasOperation trans_a,AmdBlasOperation trans_b,int m,int n,int k,float alpha,AmdDeviceAddr a,int lda,long long stride_a,AmdDeviceAddr b,int ldb,long long stride_b,float beta,AmdDeviceAddr c,int ldc,long long stride_c,int batch_count);
#ifdef __cplusplus
}
#endif

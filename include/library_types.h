#pragma once
/* Clean-room storage-type subset; values follow the public CUDA contract. */
typedef enum cudaDataType_t {
  CUDA_R_32F = 0,
  CUDA_R_64F = 1,
  CUDA_R_16F = 2,
  CUDA_R_16BF = 14
} cudaDataType;

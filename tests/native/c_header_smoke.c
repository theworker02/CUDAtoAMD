#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <amdcuda.h>
#include <driver_types.h>

int main(void) {
  CUdeviceptr pointer = 0;
  cublasHandle_t blas = 0;
  cufftHandle fft = 0;
  cudaExtent extent = make_cudaExtent(1, 2, 3);
  return (int)pointer + (blas != 0) + fft + (int)(extent.depth - 3);
}

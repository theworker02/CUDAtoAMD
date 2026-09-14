#include <cuda_runtime_api.h>

int main() {
  if (cudaGetErrorString(cudaErrorNotSupported) == nullptr) return 1;
  if (cudaMalloc(nullptr, 64) != cudaErrorInvalidValue) return 2;
  if (cudaGetLastError() != cudaErrorInvalidValue) return 3;
  static_assert(cudaErrorInvalidDevice==101 && cudaErrorInvalidResourceHandle==400 && cudaErrorNotReady==600);
  if(cudaMalloc(nullptr,1)!=cudaErrorInvalidValue)return 4;
  if(cudaFree(nullptr)!=cudaSuccess)return 5;
  if(cudaPeekAtLastError()!=cudaErrorInvalidValue)return 6;
  if(cudaGetLastError()!=cudaErrorInvalidValue || cudaPeekAtLastError()!=cudaSuccess)return 7;
  return 0;
}

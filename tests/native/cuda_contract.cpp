#include "cublas_v2.h"
#include "cuda.h"
#include "cufft.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define REQUIRE(x)                                                             \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x);              \
      return 1;                                                                \
    }                                                                          \
  } while (0)
int main(int argc, char **argv) {
  REQUIRE(argc == 2);
  int devices = 0;
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  if (!devices)
    return 77;
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  REQUIRE(cudaMalloc(nullptr, 1) == cudaErrorInvalidValue);
  REQUIRE(cudaFree(nullptr) == cudaSuccess);
  REQUIRE(cudaPeekAtLastError() == cudaErrorInvalidValue);
  CUdevice bad{};
  REQUIRE(cuDeviceGet(&bad, -1) == CUDA_ERROR_INVALID_DEVICE);
  REQUIRE(cudaPeekAtLastError() == cudaErrorInvalidValue);
  REQUIRE(cudaGetLastError() == cudaErrorInvalidValue);
  REQUIRE(cudaPeekAtLastError() == cudaSuccess);
  REQUIRE(cudaStreamQuery(reinterpret_cast<cudaStream_t>(1)) ==
          cudaErrorInvalidResourceHandle);
  cudaGetLastError();
  constexpr int n = 17;
  float *a{}, *b{}, *c{};
  REQUIRE(cudaMalloc((void **)&a, n * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc((void **)&b, n * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc((void **)&c, n * sizeof(float)) == cudaSuccess);
  std::vector<float> ha(n), hb(n), hc(n, -99);
  for (int i = 0; i < n; ++i) {
    ha[i] = i * .5f;
    hb[i] = i % 3;
  }
  REQUIRE(cudaMemcpy(a, ha.data(), n * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess);
  REQUIRE(cudaMemcpy(b, hb.data(), n * 4, cudaMemcpyHostToDevice) ==
          cudaSuccess);
  CUmodule module{};
  CUfunction function{};
  REQUIRE(cuModuleLoad(&module, argv[1]) == CUDA_SUCCESS);
  REQUIRE(cuModuleGetFunction(&function, module, "vector_add") == CUDA_SUCCESS);
  int length = n;
  void *args[] = {&a, &b, &c, &length};
  REQUIRE(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, args,
                         nullptr) == CUDA_SUCCESS);
  REQUIRE(cudaMemcpy(hc.data(), c, n * 4, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  for (int i = 0; i < n; ++i)
    REQUIRE(hc[i] == ha[i] + hb[i]);
  REQUIRE(cuModuleUnload(module) == CUDA_SUCCESS);
  REQUIRE(cuModuleUnload(module) == CUDA_ERROR_INVALID_HANDLE);
  REQUIRE(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, args,
                         nullptr) == CUDA_ERROR_INVALID_HANDLE);
  float ma[] = {1, 2, 3, 4}, mb[] = {5, 6, 7, 8}, mc[4]{}, alpha = 1, beta = 0;
  REQUIRE(cudaMemcpy(a, ma, sizeof(ma), cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(b, mb, sizeof(mb), cudaMemcpyHostToDevice) == cudaSuccess);
  cublasHandle_t blas{};
  REQUIRE(cublasCreate(&blas) == CUBLAS_STATUS_SUCCESS);
  REQUIRE(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, a, 2, b,
                      2, &beta, c, 2) == CUBLAS_STATUS_SUCCESS);
  REQUIRE(cudaMemcpy(mc, c, sizeof(mc), cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(mc[0] == 23 && mc[1] == 34 && mc[2] == 31 && mc[3] == 46);
  // Two batches, padded stride, transposed A, nontrivial alpha/beta.
  // Exact small integer fixtures avoid conflating storage conversion with GEMM.
  for (auto type : {CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF}) {
    float input[12] = {1, 2, 3, 4, 0, 0, 4, 3, 2, 1, 0, 0};
    std::uint16_t encoded[12]{};
    const std::uint16_t half[] = {0, 0x3c00, 0x4000, 0x4200, 0x4400};
    for (int i = 0; i < 12; ++i) {
      std::uint32_t bits;
      std::memcpy(&bits, &input[i], sizeof(bits));
      encoded[i] = type == CUDA_R_16F ? half[(int)input[i]]
                                      : (std::uint16_t)(bits >> 16);
    }
    const void *storage =
        type == CUDA_R_32F ? (const void *)input : (const void *)encoded;
    auto bytes = type == CUDA_R_32F ? sizeof(input) : sizeof(encoded);
    REQUIRE(cudaMemcpy(a, storage, bytes, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(b, storage, bytes, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    for (auto trans : {CUBLAS_OP_N, CUBLAS_OP_T}) {
      float output[12];
      for (float &value : output)
        value = 2;
      REQUIRE(cudaMemcpy(c, output, sizeof(output), cudaMemcpyHostToDevice) ==
              cudaSuccess);
      float scale = 2, accumulate = 3;
      REQUIRE(cublasGemmStridedBatchedEx(
                  blas, trans, CUBLAS_OP_N, 2, 2, 2, &scale, a, type, 2, 6, b,
                  type, 2, 6, &accumulate, c, CUDA_R_32F, 2, 6, 2,
                  CUBLAS_COMPUTE_32F,
                  CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
      REQUIRE(cudaMemcpy(output, c, sizeof(output), cudaMemcpyDeviceToHost) ==
              cudaSuccess);
      for (int batch = 0; batch < 2; ++batch) {
        int base = batch * 6;
        for (int col = 0; col < 2; ++col)
          for (int row = 0; row < 2; ++row) {
            float expected = 6;
            for (int q = 0; q < 2; ++q)
              expected += 2 *
                          input[base + (trans == CUBLAS_OP_N ? row + q * 2
                                                             : q + row * 2)] *
                          input[base + q + col * 2];
            REQUIRE(output[base + row + col * 2] == expected);
          }
        REQUIRE(output[base + 4] == 2 && output[base + 5] == 2);
      }
    }
    REQUIRE(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, a,
                         type, 2, b, type, 2, &beta, c, CUDA_R_32F, 2,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cudaMemcpy(mc, c, sizeof(mc), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(mc[0] == 7 && mc[1] == 10 && mc[2] == 15 && mc[3] == 22);
  }
  REQUIRE(cudaMemcpy(a, ma, sizeof(ma), cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(b, mb, sizeof(mb), cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
                                    &alpha, a, 2, 0, b, 2, 0, &beta, c, 2, 4,
                                    2) == CUBLAS_STATUS_SUCCESS);
  float repeated[8]{};
  REQUIRE(cudaMemcpy(repeated, c, sizeof(repeated), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  const float expected_repeat[] = {23, 34, 31, 46};
  for (int i = 0; i < 8; ++i)
    REQUIRE(repeated[i] == expected_repeat[i % 4]);
  REQUIRE(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, a,
                       CUDA_R_64F, 2, b, CUDA_R_64F, 2, &beta, c, CUDA_R_32F, 2,
                       CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_NOT_SUPPORTED);
  REQUIRE(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
                                    &alpha, a, 2, 4, b, 2, 4, &beta, c, 2, 1,
                                    2) == CUBLAS_STATUS_INVALID_VALUE);
  REQUIRE(cublasDestroy(blas) == CUBLAS_STATUS_SUCCESS);
  REQUIRE(cublasDestroy(blas) == CUBLAS_STATUS_NOT_INITIALIZED);
  cufftComplex impulse[4] = {{1, 0}, {0, 0}, {0, 0}, {0, 0}}, spectrum[4]{};
  REQUIRE(cudaMemcpy(a, impulse, sizeof(impulse), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  cufftHandle fft{};
  REQUIRE(cufftPlan1d(&fft, 4, CUFFT_C2C, 1) == CUFFT_SUCCESS);
  REQUIRE(cufftExecC2C(fft, (cufftComplex *)a, (cufftComplex *)c,
                       CUFFT_FORWARD) == CUFFT_SUCCESS);
  REQUIRE(cudaMemcpy(spectrum, c, sizeof(spectrum), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  for (auto v : spectrum)
    REQUIRE(v.x == 1 && v.y == 0);
  REQUIRE(cufftDestroy(fft) == CUFFT_SUCCESS);
  REQUIRE(cufftDestroy(fft) == CUFFT_INVALID_PLAN);
  cudaStream_t stream{};
  cudaEvent_t start{}, stop{};
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  REQUIRE(cudaEventCreate(&start) == cudaSuccess &&
          cudaEventCreate(&stop) == cudaSuccess);
  REQUIRE(cudaEventRecord(start, stream) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(c, 0, n * 4, stream) == cudaSuccess);
  REQUIRE(cudaEventRecord(stop, stream) == cudaSuccess);
  REQUIRE(cudaStreamWaitEvent(stream, stop, 0) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(cudaStreamQuery(stream) == cudaSuccess &&
          cudaEventQuery(stop) == cudaSuccess);
  float ms = 0;
  REQUIRE(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess && ms >= 0);
  REQUIRE(cudaEventDestroy(start) == cudaSuccess &&
          cudaEventDestroy(stop) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(cudaStreamQuery(stream) == cudaErrorInvalidResourceHandle);
  REQUIRE(cudaFree(a) == cudaSuccess && cudaFree(b) == cudaSuccess &&
          cudaFree(c) == cudaSuccess);
  std::puts("CUDA-facing Runtime + Driver kernel + cuBLAS SGEMM + cuFFT C2C "
            "verified.");
}

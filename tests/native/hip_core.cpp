#include <cuda.h>
#include <amd_blas.h>

#include <array>

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
  if (cudaSetDevice(0) != cudaSuccess) return 1;

  void* device_memory = nullptr;
  if (cudaMalloc(&device_memory, 64) != cudaSuccess) return 2;
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  const auto cleanup = [&] { if (event) cudaEventDestroy(event); if (stream) cudaStreamDestroy(stream); if (device_memory) cudaFree(device_memory); };
  if (cudaStreamCreate(&stream) != cudaSuccess) { cleanup(); return 3; }
  if (cudaEventCreate(&event) != cudaSuccess) { cleanup(); return 4; }
  if (cudaMemsetAsync(device_memory, 0, 64, stream) != cudaSuccess) { cleanup(); return 5; }
  if (cudaEventRecord(event, stream) != cudaSuccess || cudaEventSynchronize(event) != cudaSuccess) { cleanup(); return 6; }
  std::array<unsigned char, 64> host{};
  if (cudaMemcpy(host.data(), device_memory, host.size(), cudaMemcpyDeviceToHost) != cudaSuccess) { cleanup(); return 7; }
  cleanup();
  for (auto value : host) if (value != 0) return 8;
  if (cuInit(0) != CUDA_SUCCESS) return 9;
  CUdevice driver_device{};
  if (cuDeviceGet(&driver_device, 0) != CUDA_SUCCESS) return 10;
  CUdeviceptr driver_memory{};
  if (cuMemAlloc(&driver_memory, host.size()) != CUDA_SUCCESS) return 11;
  if (cuMemcpyHtoD(driver_memory, host.data(), host.size()) != CUDA_SUCCESS) return 12;
  std::array<unsigned char, 64> driver_copy{};
  if (cuMemcpyDtoH(driver_copy.data(), driver_memory, driver_copy.size()) != CUDA_SUCCESS) return 13;
  if (cuMemFree(driver_memory) != CUDA_SUCCESS) return 14;
  if (driver_copy != host) return 15;
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess || cudaGraphDestroy(graph) != cudaSuccess) return 16;
  float host_a[4] = {1, 2, 3, 4}, host_b[4] = {5, 6, 7, 8}, host_c[4] = {};
  void *a=nullptr, *b=nullptr, *c=nullptr;
  if(cudaMalloc(&a,sizeof(host_a))||cudaMalloc(&b,sizeof(host_b))||cudaMalloc(&c,sizeof(host_c))) return 17;
  if(cudaMemcpy(a,host_a,sizeof(host_a),cudaMemcpyHostToDevice)||cudaMemcpy(b,host_b,sizeof(host_b),cudaMemcpyHostToDevice)) return 18;
  AmdBlasHandle blas=nullptr; if(amdBlasCreate(&blas)!=AMD_SUCCESS) return 19;
  const auto gemm=amdBlasSgemm(blas,AMD_BLAS_OP_N,AMD_BLAS_OP_N,2,2,2,1.0F,(AmdDeviceAddr)a,2,(AmdDeviceAddr)b,2,0.0F,(AmdDeviceAddr)c,2);
  if(gemm!=AMD_SUCCESS||cudaMemcpy(host_c,c,sizeof(host_c),cudaMemcpyDeviceToHost)||host_c[0]!=23||host_c[1]!=34||host_c[2]!=31||host_c[3]!=46) return 20;
  amdBlasDestroy(blas); cudaFree(a); cudaFree(b); cudaFree(c);
  float batched_a[8] = {1,2,3,4, 2,0,0,2}, batched_b[8] = {5,6,7,8, 1,3,4,2}, batched_c[8] = {};
  if(cudaMalloc(&a,sizeof(batched_a))||cudaMalloc(&b,sizeof(batched_b))||cudaMalloc(&c,sizeof(batched_c))) return 21;
  if(cudaMemcpy(a,batched_a,sizeof(batched_a),cudaMemcpyHostToDevice)||cudaMemcpy(b,batched_b,sizeof(batched_b),cudaMemcpyHostToDevice)) return 22;
  if(amdBlasCreate(&blas)!=AMD_SUCCESS || amdBlasSgemmStridedBatched(blas,AMD_BLAS_OP_N,AMD_BLAS_OP_N,2,2,2,1.0F,(AmdDeviceAddr)a,2,4,(AmdDeviceAddr)b,2,4,0.0F,(AmdDeviceAddr)c,2,4,2)!=AMD_SUCCESS) return 23;
  if(cudaMemcpy(batched_c,c,sizeof(batched_c),cudaMemcpyDeviceToHost)||batched_c[0]!=23||batched_c[3]!=46||batched_c[4]!=2||batched_c[5]!=6||batched_c[6]!=8||batched_c[7]!=4) return 24;
  amdBlasDestroy(blas); cudaFree(a); cudaFree(b); cudaFree(c);
  return 0;
}

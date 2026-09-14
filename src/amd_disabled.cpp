#include "amd_blas.h"
AmdResult amdBlasCreateOnDevice(AmdDevice, AmdBlasHandle* handle) {
  if(!handle)return AMD_ERROR_INVALID_VALUE;
  *handle=nullptr;return AMD_ERROR_NOT_INITIALIZED;
}
AmdResult amdBlasGemmStridedBatchedEx(AmdBlasHandle,AmdStream,AmdBlasDataType,int,int,int,int,int,float,AmdDeviceAddr,int,long long,AmdDeviceAddr,int,long long,float,AmdDeviceAddr,int,long long,int) {return AMD_ERROR_NOT_INITIALIZED;}
AmdResult amdEventCreateOnDevice(AmdDevice, AmdEvent* event) {
  if (!event) return AMD_ERROR_INVALID_VALUE;
  *event = nullptr;
  return AMD_ERROR_NOT_INITIALIZED;
}
#include "amd_runtime.h"

// Backend-free builds retain linkable APIs, but never pretend to execute work.
extern "C" AmdResult amdInit(uint32_t flags) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdGetDeviceCount(int *count) {
  if (!count)
    return AMD_ERROR_INVALID_VALUE;
  *count = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdGetDevice(AmdDevice *device, int index) {
  if (!device)
    return AMD_ERROR_INVALID_VALUE;
  *device = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdDeviceDestroy(AmdDevice device) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdGetDeviceProperties(AmdDevice device,
                                            AmdDeviceProps *props) {
  if (!props)
    return AMD_ERROR_INVALID_VALUE;
  *props = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemPoolCreate(AmdDevice device, size_t initial_capacity,
                                      AmdMemPool *pool) {
  if (!pool)
    return AMD_ERROR_INVALID_VALUE;
  *pool = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemPoolDestroy(AmdMemPool pool) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemAllocAsync(AmdDeviceAddr *dptr, size_t bytes,
                                      AmdMemPool pool, AmdStream stream) {
  if (!dptr)
    return AMD_ERROR_INVALID_VALUE;
  *dptr = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemFreeAsync(AmdDeviceAddr dptr, AmdMemPool pool,
                                     AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemcpyHtoDAsync(AmdDeviceAddr dst, const void *src,
                                        size_t bytes, AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemcpyDtoHAsync(void *dst, AmdDeviceAddr src,
                                        size_t bytes, AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemcpyDtoDAsync(AmdDeviceAddr dst, AmdDeviceAddr src,
                                        size_t bytes, AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemHostRegister(void *ptr, size_t bytes) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdMemHostUnregister(void *ptr) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdStreamCreate(AmdDevice device, AmdStream *stream) {
  if (!stream)
    return AMD_ERROR_INVALID_VALUE;
  *stream = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdStreamDestroy(AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdStreamSynchronize(AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdStreamQuery(AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdEventCreate(AmdEvent *event) {
  if (!event)
    return AMD_ERROR_INVALID_VALUE;
  *event = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdEventDestroy(AmdEvent event) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdEventRecord(AmdEvent event, AmdStream stream) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdStreamWaitEvent(AmdStream stream, AmdEvent event) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdEventElapsedTime(float *ms, AmdEvent start,
                                         AmdEvent stop) {
  if (!ms)
    return AMD_ERROR_INVALID_VALUE;
  *ms = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdModuleLoadFile(AmdDevice device, const char *path,
                                       AmdModule *module) {
  if (!module)
    return AMD_ERROR_INVALID_VALUE;
  *module = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdModuleLoadData(AmdDevice device,
                                       const void *image_bytes, size_t size,
                                       AmdModule *module) {
  if (!module)
    return AMD_ERROR_INVALID_VALUE;
  *module = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdModuleUnload(AmdModule module) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdModuleGetFunction(AmdFunction *function,
                                          AmdModule module, const char *name) {
  if (!function)
    return AMD_ERROR_INVALID_VALUE;
  *function = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdLaunchKernel(AmdFunction function, AmdDim3 grid,
                                     AmdDim3 block, uint32_t shared,
                                     AmdStream stream, void **args) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdBlasCreate(AmdBlasHandle *handle) {
  if (!handle)
    return AMD_ERROR_INVALID_VALUE;
  *handle = {};
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdBlasDestroy(AmdBlasHandle handle) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdBlasSgemm(AmdBlasHandle handle,
                                  AmdBlasOperation trans_a,
                                  AmdBlasOperation trans_b, int m, int n, int k,
                                  float alpha, AmdDeviceAddr a, int lda,
                                  AmdDeviceAddr b, int ldb, float beta,
                                  AmdDeviceAddr c, int ldc) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" AmdResult amdBlasSgemmStridedBatched(
    AmdBlasHandle handle, AmdBlasOperation trans_a, AmdBlasOperation trans_b,
    int m, int n, int k, float alpha, AmdDeviceAddr a, int lda,
    long long stride_a, AmdDeviceAddr b, int ldb, long long stride_b,
    float beta, AmdDeviceAddr c, int ldc, long long stride_c, int batch_count) {
  return AMD_ERROR_NOT_INITIALIZED;
}
extern "C" const char *amdGetErrorString(AmdResult r) {
  return r == AMD_SUCCESS                 ? "success"
         : r == AMD_ERROR_NOT_INITIALIZED ? "HIP backend disabled at build time"
                                          : "AMD runtime error";
}

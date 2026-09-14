#ifndef AMD_RUNTIME_H
#define AMD_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

/* Native handle type/lifetime is validated; raw data pointers are caller-owned.
 * Tombstone metadata is retained until process exit to prevent stale reuse.
 * Lifetime contract: handles must be live and created by this runtime. Function
 * handles are borrowed from modules and invalid after unload. Destruction must
 * not race any operation on the object. Pool allocation/free is serialized.
 * Pool initial_capacity is a retention threshold, not preallocated VRAM.
 * Pool destruction rejects live allocations and waits for pending device work.
 * Module unload waits for device work. Host buffers must survive async copies.
 * LoadData accepts bounded ELF64 AMDGPU objects, not PTX or offload bundles;
 * validation is not a sandbox: only load trusted code objects. */

#ifdef __cplusplus
extern "C" {
#endif
typedef struct AmdDevice_t* AmdDevice; typedef struct AmdStream_t* AmdStream; typedef struct AmdEvent_t* AmdEvent; typedef struct AmdModule_t* AmdModule; typedef struct AmdFunction_t* AmdFunction; typedef struct AmdMemPool_t* AmdMemPool;
typedef uint64_t AmdDeviceAddr;
typedef enum AmdResult { AMD_SUCCESS=0, AMD_ERROR_INVALID_VALUE=1, AMD_ERROR_OUT_OF_MEMORY=2, AMD_ERROR_NOT_INITIALIZED=3, AMD_ERROR_DEVICE_NOT_FOUND=4, AMD_ERROR_INVALID_HANDLE=5, AMD_ERROR_MODULE_LOAD_FAILED=6, AMD_ERROR_SYMBOL_NOT_FOUND=7, AMD_ERROR_LAUNCH_FAILED=8, AMD_ERROR_NOT_READY=9, AMD_ERROR_UNKNOWN=999 } AmdResult;
typedef enum AmdWavefrontSize { AMD_WAVE_SIZE_32=32, AMD_WAVE_SIZE_64=64 } AmdWavefrontSize;
typedef struct AmdDeviceProps { char name[256]; char gcn_arch[64]; size_t total_vram_bytes; uint32_t compute_units; uint32_t max_threads_per_block; AmdWavefrontSize native_wave_size; int supports_wmma; int is_cdna; } AmdDeviceProps;
typedef struct AmdDim3 { uint32_t x, y, z; } AmdDim3;
#ifdef __cplusplus
#define AMD_DIM3(x,y,z) AmdDim3{(x),(y),(z)}
#else
#define AMD_DIM3(x,y,z) ((AmdDim3){(x),(y),(z)})
#endif
AmdResult amdEventCreateOnDevice(AmdDevice device, AmdEvent* event);
AmdResult amdInit(uint32_t flags); AmdResult amdGetDeviceCount(int* count); AmdResult amdGetDevice(AmdDevice* device, int index); AmdResult amdDeviceDestroy(AmdDevice device); AmdResult amdGetDeviceProperties(AmdDevice device, AmdDeviceProps* props);
AmdResult amdMemPoolCreate(AmdDevice device, size_t initial_capacity, AmdMemPool* pool); AmdResult amdMemPoolDestroy(AmdMemPool pool); AmdResult amdMemAllocAsync(AmdDeviceAddr* dptr, size_t bytes, AmdMemPool pool, AmdStream stream); AmdResult amdMemFreeAsync(AmdDeviceAddr dptr, AmdMemPool pool, AmdStream stream);
AmdResult amdMemcpyHtoDAsync(AmdDeviceAddr dst,const void* src,size_t bytes,AmdStream stream); AmdResult amdMemcpyDtoHAsync(void* dst,AmdDeviceAddr src,size_t bytes,AmdStream stream); AmdResult amdMemcpyDtoDAsync(AmdDeviceAddr dst,AmdDeviceAddr src,size_t bytes,AmdStream stream); AmdResult amdMemHostRegister(void* ptr,size_t bytes); AmdResult amdMemHostUnregister(void* ptr);
AmdResult amdStreamCreate(AmdDevice device,AmdStream* stream); AmdResult amdStreamDestroy(AmdStream stream); AmdResult amdStreamSynchronize(AmdStream stream); AmdResult amdStreamQuery(AmdStream stream); AmdResult amdEventCreate(AmdEvent* event); AmdResult amdEventDestroy(AmdEvent event); AmdResult amdEventRecord(AmdEvent event,AmdStream stream); AmdResult amdStreamWaitEvent(AmdStream stream,AmdEvent event); AmdResult amdEventElapsedTime(float* ms,AmdEvent start,AmdEvent stop);
AmdResult amdModuleLoadFile(AmdDevice device,const char* path,AmdModule* module); AmdResult amdModuleLoadData(AmdDevice device,const void* image_bytes,size_t size,AmdModule* module); AmdResult amdModuleUnload(AmdModule module); AmdResult amdModuleGetFunction(AmdFunction* function,AmdModule module,const char* name); AmdResult amdLaunchKernel(AmdFunction function,AmdDim3 grid,AmdDim3 block,uint32_t shared,AmdStream stream,void** args); const char* amdGetErrorString(AmdResult result);
#ifdef __cplusplus
}
#endif
#endif

#pragma once

#include "cuda_runtime_api.h"

#include <stdint.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef uintptr_t CUdeviceptr;
enum { CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1, CUDA_ERROR_OUT_OF_MEMORY = 2, CUDA_ERROR_INVALID_DEVICE = 101, CUDA_ERROR_NOT_SUPPORTED = 801, CUDA_ERROR_UNKNOWN = 999 };
enum { CUDA_ERROR_INVALID_CONTEXT=201, CUDA_ERROR_INVALID_HANDLE=400, CUDA_ERROR_NOT_FOUND=500, CUDA_ERROR_NOT_READY=600 };

#ifdef __cplusplus
extern "C" {
#endif
COMPATCUDA_API CUresult cuInit(unsigned int flags);
COMPATCUDA_API CUresult cuDeviceGetCount(int* count);
COMPATCUDA_API CUresult cuDeviceGet(CUdevice* device, int ordinal);
COMPATCUDA_API CUresult cuCtxCreate(CUcontext* context, unsigned int flags, CUdevice device);
COMPATCUDA_API CUresult cuCtxDestroy(CUcontext context);
COMPATCUDA_API CUresult cuMemAlloc(CUdeviceptr* pointer, size_t bytes);
COMPATCUDA_API CUresult cuMemFree(CUdeviceptr pointer);
COMPATCUDA_API CUresult cuMemcpyHtoD(CUdeviceptr destination, const void* source, size_t bytes);
COMPATCUDA_API CUresult cuMemcpyDtoH(void* destination, CUdeviceptr source, size_t bytes);
COMPATCUDA_API CUresult cuModuleLoad(CUmodule* module, const char* path);
COMPATCUDA_API CUresult cuModuleUnload(CUmodule module);
COMPATCUDA_API CUresult cuModuleGetFunction(CUfunction* function,CUmodule module,const char* name);
COMPATCUDA_API CUresult cuLaunchKernel(CUfunction function,unsigned int gx,unsigned int gy,unsigned int gz,unsigned int bx,unsigned int by,unsigned int bz,unsigned int shared,cudaStream_t stream,void** arguments,void** extra);
#ifdef __cplusplus
}
#endif

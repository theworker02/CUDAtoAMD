#pragma once

#include "cuda_runtime_api.h"

/* Explicit native-AMD module API. It accepts only AMDGPU/HIP loadable code objects. */
#ifdef __cplusplus
struct AMDCUModule;
struct AMDCUFunction;
#else
typedef struct AMDCUModule AMDCUModule;
typedef struct AMDCUFunction AMDCUFunction;
#endif
typedef AMDCUModule* AMDCUmodule;
typedef AMDCUFunction* AMDCUfunction;

#ifdef __cplusplus
extern "C" {
#endif
COMPATCUDA_API cudaError_t amdcuModuleLoad(AMDCUmodule* module, const char* hsaco_path);
COMPATCUDA_API cudaError_t amdcuModuleUnload(AMDCUmodule module);
COMPATCUDA_API cudaError_t amdcuModuleGetFunction(AMDCUfunction* function, AMDCUmodule module, const char* name);
COMPATCUDA_API cudaError_t amdcuLaunchKernel(AMDCUfunction function, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z, unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int shared_bytes, cudaStream_t stream, void** arguments);
COMPATCUDA_API cudaError_t amdcuMallocAsync(void** pointer, size_t bytes, cudaStream_t stream);
COMPATCUDA_API cudaError_t amdcuFreeAsync(void* pointer, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

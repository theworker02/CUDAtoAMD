#pragma once

/* Independently implemented, source-level CUDA Runtime subset. */
#include <stddef.h>

#if defined(_WIN32)
  #if defined(COMPATCUDA_BUILDING_LIBRARY)
    #define COMPATCUDA_API __declspec(dllexport)
  #else
    #define COMPATCUDA_API __declspec(dllimport)
  #endif
#else
  #define COMPATCUDA_API __attribute__((visibility("default")))
#endif

typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorInvalidDevice = 101,
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorNotReady = 600,
  cudaErrorNotSupported = 801,
  cudaErrorUnknown = 999
} cudaError_t;

typedef enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
} cudaMemcpyKind;

#ifdef __cplusplus
struct CompatStream;
struct CompatEvent;
struct CompatGraph;
#else
typedef struct CompatStream CompatStream;
typedef struct CompatEvent CompatEvent;
typedef struct CompatGraph CompatGraph;
#endif
typedef CompatStream* cudaStream_t;
typedef CompatEvent* cudaEvent_t;
typedef CompatGraph* cudaGraph_t;

#ifdef __cplusplus
extern "C" {
#endif
COMPATCUDA_API cudaError_t cudaGetDeviceCount(int* count);
COMPATCUDA_API cudaError_t cudaSetDevice(int device);
COMPATCUDA_API cudaError_t cudaGetDevice(int* device);
COMPATCUDA_API cudaError_t cudaDeviceSynchronize(void);
COMPATCUDA_API cudaError_t cudaGetLastError(void);
COMPATCUDA_API cudaError_t cudaPeekAtLastError(void);
COMPATCUDA_API cudaError_t cudaStreamQuery(cudaStream_t stream);
COMPATCUDA_API cudaError_t cudaStreamWaitEvent(cudaStream_t stream,cudaEvent_t event,unsigned int flags);
COMPATCUDA_API cudaError_t cudaEventQuery(cudaEvent_t event);
COMPATCUDA_API cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t start,cudaEvent_t stop);
COMPATCUDA_API const char* cudaGetErrorString(cudaError_t error);
COMPATCUDA_API cudaError_t cudaMalloc(void** pointer, size_t bytes);
COMPATCUDA_API cudaError_t cudaFree(void* pointer);
COMPATCUDA_API cudaError_t cudaMemcpy(void* destination, const void* source, size_t bytes, cudaMemcpyKind kind);
COMPATCUDA_API cudaError_t cudaMemcpyAsync(void* destination, const void* source, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream);
COMPATCUDA_API cudaError_t cudaMemset(void* pointer, int value, size_t bytes);
COMPATCUDA_API cudaError_t cudaMemsetAsync(void* pointer, int value, size_t bytes, cudaStream_t stream);
COMPATCUDA_API cudaError_t cudaStreamCreate(cudaStream_t* stream);
COMPATCUDA_API cudaError_t cudaStreamDestroy(cudaStream_t stream);
COMPATCUDA_API cudaError_t cudaStreamSynchronize(cudaStream_t stream);
COMPATCUDA_API cudaError_t cudaEventCreate(cudaEvent_t* event);
COMPATCUDA_API cudaError_t cudaEventDestroy(cudaEvent_t event);
COMPATCUDA_API cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
COMPATCUDA_API cudaError_t cudaEventSynchronize(cudaEvent_t event);
COMPATCUDA_API cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags);
COMPATCUDA_API cudaError_t cudaGraphDestroy(cudaGraph_t graph);
#ifdef __cplusplus
}
#endif

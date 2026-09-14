# Compatibility Matrix

Generated from `compatibility/cuda_api.json`; this is an interface inventory, not an application-success percentage.

## Inventory statistics

| Status | Interfaces |
| --- | ---: |
| ADAPTED | 12 |
| DIRECT | 10 |
| NVIDIA_SPECIFIC | 1 |
| PARTIAL | 6 |
| UNSUPPORTED | 1 |
| **Total tracked interfaces** | **30** |

A `DIRECT` entry has a documented HIP/ROCm analogue; it is not an assertion that ABI-level CUDA drop-in execution already exists. `PARTIAL` and `ADAPTED` entries must be validated by conformance tests before runtime support is claimed.

## Atomics

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| __nvvm_atom | device | NVIDIA_SPECIFIC | — | none | none |

## Blas

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cublasCreate | cublas | ADAPTED | hipblasCreate | partial | library adapter |
| cublasSgemm | cublas | ADAPTED | hipblasSgemm | partial | library adapter |

## Collectives

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| ncclCommInitRank | nccl | PARTIAL | rcclCommInitRank | partial | library adapter |

## Cooperative Operations

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudaLaunchCooperativeKernel | runtime | PARTIAL | hipLaunchCooperativeKernel | partial | capability validation |

## Device Management

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudaDeviceSynchronize | runtime | DIRECT | hipDeviceSynchronize | full | direct |
| cudaGetDeviceCount | runtime | DIRECT | hipGetDeviceCount | full | direct |
| cudaGetDeviceProperties | runtime | ADAPTED | hipGetDeviceProperties | partial | capability virtualization |
| cudaSetDevice | runtime | DIRECT | hipSetDevice | full | direct |

## Dnn

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudnnCreate | cudnn | PARTIAL | miopenCreate | partial | library adapter |

## Events

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudaEventCreate | runtime | DIRECT | hipEventCreate | full | direct |
| cudaEventElapsedTime | runtime | ADAPTED | hipEventElapsedTime | partial | error normalization |
| cudaEventRecord | runtime | DIRECT | hipEventRecord | full | direct |

## Fft

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cufftExecC2C | cufft | ADAPTED | hipfftExecC2C | partial | library adapter |
| cufftPlan1d | cufft | ADAPTED | hipfftPlan1d | partial | library adapter |

## Graphs

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudaGraphLaunch | runtime | PARTIAL | hipGraphLaunch | partial | runtime shim |

## Ipc

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudaIpcOpenMemHandle | runtime | UNSUPPORTED | — | none | none |

## Memory

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudaFree | runtime | DIRECT | hipFree | full | direct |
| cudaMalloc | runtime | DIRECT | hipMalloc | full | direct |
| cudaMallocManaged | runtime | PARTIAL | hipMallocManaged | partial | runtime shim |
| cudaMemcpy | runtime | ADAPTED | hipMemcpy | partial | argument translation |
| cudaMemcpyAsync | runtime | ADAPTED | hipMemcpyAsync | partial | argument translation |
| cudaMemset | runtime | DIRECT | hipMemset | full | direct |

## Random

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| curandCreateGenerator | curand | ADAPTED | hiprandCreateGenerator | partial | library adapter |

## Runtime Compilation

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| nvrtcCompileProgram | nvrtc | ADAPTED | hiprtcCompileProgram | partial | source translation |
| nvrtcCreateProgram | nvrtc | ADAPTED | hiprtcCreateProgram | partial | source translation |

## Sparse

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cusparseCreate | cusparse | ADAPTED | hipsparseCreate | partial | library adapter |

## Streams

| CUDA symbol | Namespace | Status | AMD target | Semantics | Strategy |
| --- | --- | --- | --- | --- | --- |
| cudaStreamCreate | runtime | DIRECT | hipStreamCreate | full | direct |
| cudaStreamCreateWithPriority | runtime | PARTIAL | hipStreamCreateWithPriority | partial | argument translation |
| cudaStreamSynchronize | runtime | DIRECT | hipStreamSynchronize | full | direct |

# CUDA-oriented source workflow (0.18.0)

This release verifies a narrow but real path from CUDA-syntax kernel source to AMD execution. It does not promise full CUDA parity. The host application links to `compatcuda.lib` / `compatcuda.dll`, not NVIDIA libraries. No drivers are replaced and no imports are injected.

## Run the acceptance workload

Build the HIP-enabled project as described in the [local guide](local-developer-guide.md), with hipFFT installed and enabled. On the verified RX 7800 XT / Windows HIP SDK 7.1 host, from the repository root:

```powershell
python -m compat code-object examples/cuda_vector.cu -o compiled-vector.hsaco --arch gfx1101
python -m compat run build-hip-check/compatcuda_contract.exe compiled-vector.hsaco
ctest --test-dir build-hip-check --output-on-failure
python -m unittest discover -s tests -v
```

Choose a new output filename for another compilation: the compiler deliberately refuses to overwrite files. Use your GPU's actual gfx target; other architectures and Linux have not been verified. The contract executable is built only when HIP tests and hipFFT are enabled.

`examples/cuda_vector.cu` remains unchanged. HIP-Clang parses its supported CUDA-style kernel syntax, with the HIP device header supplied by the compiler command. The command produces an unbundled ELF code object, not a host executable. It is distinct from the older, unverified HIPIFY-based `compat cc` workflow.

`tests/native/cuda_contract.cpp` is the runnable host example: it uses Runtime allocations/copies, Driver module lookup/launch, cuBLAS SGEMM and cuFFT C2C, then compares every kernel/GEMM/FFT output against expected results. It also tests stale handles, stream/event operations and last-error behavior. The host source does not call HIP.

## Implemented source-facing subset

| Boundary | Implemented scope | Important restriction |
| --- | --- | --- |
| Compiler | CUDA-syntax kernel subset accepted by HIP-Clang; separate experimental straight-line PTX lowering | No general PTX, cubin or NVVM translation; no automatic host application conversion |
| Runtime | Existing allocations/copies/devices plus stream query/wait and event query/timing | Raw pointer ranges are not validated |
| Driver | Contexts, native module loading, function lookup and argument-array launch | Native AMD modules only; launch `extra` buffers unsupported |
| `cublas_v2.h` | Create/destroy, SGEMM, strided-batched SGEMM, GemmEx and GemmStridedBatchedEx | Default stream, host scalar pointers; no BLASLt or complete cuBLAS ABI |
| `cufft.h` | Plan1d, ExecC2C and Destroy; batched complex FP32 | Default stream, optional hipFFT; no full cuFFT ABI |

These CUDA-facing host headers are currently tested as C++ headers. Native `amd_*.h` APIs remain the explicit C interface. Library symbols are exported from compatcuda, not DLLs named after NVIDIA libraries. Use the native adapters when you need stream-bound mixed-precision GEMM or additional native functionality.

Implemented wrapper handles use type/lifetime checks and retained tombstones. Module unload invalidates its function handles and synchronizes conservatively. This costs host metadata/locking; it is not zero-overhead. Host/data-pointer validity and cross-stream ordering remain caller obligations. Do not mix the legacy facade into native graph capture: those calls are rejected while capture is active.

Runtime error numbers and peek/reset behavior follow the documented [CUDA error types](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html) and [error handling contract](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html). Passing this suite establishes only its tested behavior, not comprehensive API conformance.

## Still not implemented

### Mixed-precision GEMM contract

The Ex entry points support matching `CUDA_R_32F`, `CUDA_R_16F` or `CUDA_R_16BF` A/B storage, `CUDA_R_32F` C storage, `CUBLAS_COMPUTE_32F`, and `CUBLAS_GEMM_DEFAULT`. Alpha/beta point to host float values. No automatic precision downgrade occurs. Other combinations return `CUBLAS_STATUS_NOT_SUPPORTED`. Strides use elements, not bytes; output batches must not overlap. Zero input strides share an input matrix across batches. For real inputs conjugate transpose is equivalent to transpose.

These are a subset of the documented [cuBLAS GEMM contracts](https://docs.nvidia.com/cuda/archive/12.0.0/cublas/index.html), independently routed through the native adapter. The numerical acceptance workload covers FP32/FP16/BF16 inputs, transposed A, padded two-batch layouts, alpha/beta and single Ex GEMM. Storage conversion helpers, device scalar pointer mode, non-default stream binding and algorithm selection are not implemented in this CUDA facade.

### Remaining boundaries

Full CUDA Runtime/Driver API coverage, general PTX/NVVM/cubin translation, complete CUDA graph semantics, broad library parity, transparent CUDA-wheel execution, automatic framework interception and raw-pointer memory safety remain substantial separate work. A [small explicit PTX subset](ptx-subset.md) and [explicit PyTorch inference adapter](framework-integration.md) are implemented and tested. The source inventory database lists potential mappings; it is not a measured percentage of implemented CUDA support.

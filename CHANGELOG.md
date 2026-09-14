# Changelog

## 1.3.0

- Added an original CUDAtoAMD project logo for repository, package and documentation use.
- Added a dependency-free GitHub Pages site with practical setup, native-execution, source-compiler and wheel-safety guidance.
- Added a GitHub Actions Pages deployment workflow, a concise setup guide and GitHub-first release-distribution guidance. The workflow is ready to run after this directory is committed and pushed to a GitHub repository; it does not publish a site from a local checkout.
- Added structural tests that keep the static site, logo and Pages deployment configuration present in future releases.
- Promoted the project version to 1.3.0. The release still represents an experimental, explicit AMD-native compatibility subset rather than full CUDA or transparent CUDA-wheel parity.

## 1.2.0

- Added `compat init`, which loads the built native runtime, executes `amdInit`, selects a real AMD HIP device and reports its properties. An optional native softmax numerical self-test proves device execution.
- Added `compat toolchain`, reporting HIPCC/HIPIFY availability and the actual CUDA-facing source/PTX compiler paths that target AMD hardware.
- Added initialization documentation and a real-device test validating initialization against the verified RX 7800 XT / gfx1101 host.
- Promoted the project version to 1.2.0 for the stable documented initialization, toolchain-reporting, runtime, compiler-subset and release-validation workflows. This version does not claim full CUDA API, PTX/NVVM, or transparent CUDA-wheel parity.

## 0.36.0

- Added SHA-256 fingerprints and distribution metadata to wheel preflight reports.
- Added `compat python --wheel-sha256` to block a wheel when it does not exactly match the reviewed artifact before the target Python script runs.
- Updated the release recipe to use an explicit `Release` CMake configuration.
- Added tests for hash mismatch refusal before script execution. Version advances three minor increments for this completed admission-control batch at the user's request.

## 0.33.0

- Added bounded, read-only `compat wheel-doctor` inspection for Python wheel metadata, native extensions, CUDA device artifacts, common CUDA import strings and NVIDIA runtime dependencies.
- Added guarded `compat python`: an explicit ROCm-enabled Python launcher that can set the native runtime library path and refuses preflight-blocked/unverified CUDA-native wheels before launching.
- Added wheel fixtures covering CUDA-linked extension blocking, pure-Python handling, malformed archives and refusal before target-script execution.
- Added [CUDA wheel execution documentation](docs/cuda-wheel-execution.md) and made it a required release-gate document.
- Version advances three minor increments for this completed batch at the user's request. This is a safe admission/diagnostic layer, not CUDA DLL interception, device spoofing or transparent CUDA-wheel support.

## 0.30.0

- Added CMake installation rules for the runtime library, public headers, license, legal notes and security policy. Tests are now off by default for consumer builds and explicitly enabled for validation builds.
- Made the CUDA Runtime, Driver, and explicit AMD module public headers usable from C as well as C++; added a compiled/linkable C-header smoke test.
- Added `compat release-check`, which verifies release version alignment, required documentation, CMake install support and C-header coverage without making unverified GPU claims.
- Added [RELEASING.md](RELEASING.md) with a reproducible clean-build, test, staged-install, metadata and wheel validation sequence.
- Version advances three minor increments for this release-readiness batch at the user's request. It releases only the documented experimental subset, not CUDA parity.

## 0.27.0

- Extended PTX lowering with round-to-nearest FP32 multiply and fused multiply-add (`mul.rn.f32`, `fma.rn.f32`) plus u32-to-FP32 conversion.
- Added a bounds-safe FP32 SAXPY fixture and GPU numerical test, covering scalar FP32 parameters, 192 threads over 129 elements, FP32 global loads/stores and fused arithmetic.
- Version advances three minor increments for this completed compiler batch at the user's request. General PTX/NVVM, tensor instructions, memory-model semantics and transparent framework execution are still not implemented.

## 0.24.0

- Extended PTX lowering with u32 global loads/stores, `mul.wide.u32`, subtraction, bitwise u32 operations, u32/u64 conversions, additional unsigned comparisons, and negated predicate branches.
- Added a numerical GPU test for a bounds-safe 32-bit affine transform that exercises integer parameter handling, address calculation, wide multiplication and u32 global memory.
- Made `AmdInferenceModule` inherit from `torch.nn.Module` when PyTorch is installed, preserving standard module call semantics while retaining explicit CPU float32 staging and fail-closed autograd behavior.
- Version advances three minor increments for this completed batch at the user's request. This is still a bounded compiler/backend slice, not general PTX/NVVM, CUDA device integration, or transparent CUDA framework execution.

## 0.21.0

- Extended the fail-closed PTX subset compiler with predicate registers, `setp.lt.u32`, and forward conditional/unconditional branches. Added a bounds-safe PTX vector-add example and GPU test with 192 launched threads over 179 elements.
- Added `AmdInferenceModule`, a closeable explicit PyTorch-style adapter for local FP32 RMSNorm, softmax, SwiGLU and RoPE execution through the native AMD session.
- Added `compat framework-doctor` with JSON output identifying exactly what the framework integration does and does not support.
- The adapter deliberately preserves CPU float32 staging and rejects autograd; it is not `torch.cuda`, a custom PyTorch wheel, or CUDA binary interception. General PTX/NVVM and framework execution remain separate, unimplemented work.
- Version advances three minor increments for this completed batch at the user's request.

## 0.18.0

- Added fail-closed straight-line PTX parsing, register type/initialization checks, and explicit PTX-to-HIP lowering for a small documented instruction subset.
- Added `compat ptx-object` ahead-of-time compilation to native AMD ELF code objects. Existing source and output files are preserved; malformed or unsupported PTX is rejected before compiler invocation.
- Added original PTX vector-add fixture, parser rejection tests and numerical GPU execution tests for 32-thread and 192-thread launches.
- Version advances three minor increments for this batch at the user's request. This does not imply broad PTX conformance: NVVM, branches, atomics, tensor instructions, runtime JIT interception and transparent framework execution remain unsupported.

## 0.15.0

- Added CUDA-facing `cublasGemmEx`, `cublasGemmStridedBatchedEx` and `cublasSgemmStridedBatched`, backed by the existing native math adapter.
- Added `library_types.h` with documented storage-type values. Ex operations accept equal FP32/FP16/BF16 input types, FP32 output/scalars/compute and the default algorithm. Unsupported combinations return NOT_SUPPORTED rather than being silently cast.
- Expanded numerical CUDA contract coverage: both transpose modes, two batches, padded strides, alpha/beta, unchanged output padding, single Ex operations, and zero-stride shared inputs. Unsupported FP64 and overlapping output batches are rejected.
- Scope remains default-stream, host-scalar source compatibility; no PTX/NVVM translation, full cuBLAS ABI or transparent CUDA framework execution.

## 0.14.0

- Added `compat code-object`: compile a supported CUDA-syntax kernel subset directly with HIP-Clang to an unbundled AMD code object without changing the input source. Outputs never overwrite existing files.
- Added Driver module function lookup and kernel launch, with checked context/module/function lifetimes and child-function retirement on unload.
- Added checked legacy stream/event/graph/module handles, stream query/wait and event query/timing. Corrected CUDA error numbers and sticky last-error/peek/reset behavior; Driver errors no longer modify the Runtime last-error slot.
- Added CUDA-named cuBLAS FP32 SGEMM and cuFFT batched 1D complex FP32 facades in compatcuda, using existing native adapters. These are default-stream source-facing subsets, not vendor ABI replacement DLLs.
- Added an end-to-end Runtime + Driver kernel + BLAS + FFT numerical contract test and a real `.cu` compiler-to-execution test.
- Verified Windows HIP build (5 CTests), backend-disabled build (3 CTests), and FFT-disabled compilation. Full CUDA parity, PTX/NVVM translation and automatic framework binary compatibility remain unimplemented.

## 0.13.0

- Added native AMD opaque-handle type/lifetime validation and serialized destruction; bounded retained tombstones prevent stale address reuse.
- Added HIP single-stream graph capture, instantiation, replay and cancellation with conservative resource lifetime pins.
- Added portable FP32 RMSNorm, stable softmax, SwiGLU and interleaved RoPE code objects with NumPy comparisons.
- Added an explicit CPU-staged NumPy/PyTorch inference Session, with gradient rejection and checked data layouts.
- Added a local developer guide, real-device doctor mode and numerical demo CLI.
- Scope is not full CUDA graph parity or transparent framework interception. Legacy CUDA handles and raw data pointers do not gain the native handle guarantees.

## 0.12.0

- Added device-selected, stream-bound FP16/BF16/FP32 strided-batched GEMM with FP32 accumulation/output, transpose support and layout/stride validation.
- Added optional hipFFT-backed contiguous batched 1D complex FP32 FFT with in-place execution and unnormalized inverse.
- Added amd_fft.h and Python Blas/FFTPlan wrappers with checked buffer extents.
- Verified 12 GEMM dtype/transpose cases with padded leading dimensions, nontrivial alpha/beta and two batches, plus length-7/8 forward DFT comparisons and inverse round trips.
- Missing/disabled hipFFT keeps linkable deterministic stubs. No vendor libraries are bundled or hardware spoofed.

## 0.11.0

- Added installable Python native bindings with lazy, explicit runtime loading and retained Windows DLL directory cookies, including versioned HIP DLL discovery.
- Implemented file/data module loading, typed scalar/buffer kernel arguments, host/device copies, buffers, events and checked resource lifetimes.
- Added amdEventCreateOnDevice to the native and backend-free APIs.
- Added real Python GPU conformance tests and a runnable vector-add example.
- Included the compatibility database and CFFI declarations in the Python wheel; CFFI is an optional native extra. Native DLLs and vendor dependencies are not bundled.

## 0.10.0

- Restored the HIP-disabled build with linkable, deterministic native API stubs.
- Added compiled AMD ELF kernel correctness coverage on the configured GPU target.
- Native modules now own their function wrappers; module unload waits for device work.
- Stream/module creation selects the requested device and restores the previous device.
- Pools track allocation ownership, reject wrong-pool/double frees and reject destruction with live allocations.
- C++ buffers retain pool/stream owners across wrapper moves and provide nonthrowing destruction plus explicit checked close.
- Validated native ELF file ranges before loading memory images; PTX and bundled inputs are not accepted.
- Updated readiness reporting. Python module/copy bindings, mixed precision, FFT, neural operators and framework support remain unfinished.

## 0.9.0

- Added hipBLAS-backed strided-batched FP32 SGEMM with GPU numerical conformance coverage.

## 0.8.0

- Completed Python device ownership and added event/stream dependency bindings.

## 0.7.0

- Fixed Windows ROCm DLL discovery for CTest executables linked to hipBLAS.

## 0.6.0

- Added the native hipBLAS-backed FP32 SGEMM adapter and GPU conformance test.

## 0.5.0

- Completed native C++ device lifecycle ownership.

## 0.4.0

- Added C++20 RAII wrappers and Python CFFI bindings for the explicit native AMD runtime.

## 0.3.0

- Added the explicit `amd_runtime.h` native AMD C API for devices, memory pools, streams, events, host registration, and AMDGPU code objects.

## 0.2.0

- Expanded the clean-room CUDA-style header surface with vector, driver, channel, texture, surface, FP16, and BF16 types.
- Added native AMD `.hsaco` module and async memory-pool APIs.

## 0.1.0

- Added compatibility database, matrix generator, static analyzer, diagnostics baseline, and foundational specifications.
- No HIP execution backend or GPU compatibility claim.
- Added `compatcuda` CMake ABI target, CUDA-facing declaration subset, generated ABI inventory, and binary-format reporting.
- Replaced core fail-closed stubs with HIP-backed device, allocation, copy, stream, and event dispatch when `COMPATCUDA_ENABLE_HIP=ON`.
- Added the Phase III AI compatibility manifest and `compat profile ai` readiness report.
- Added HIP-backed Driver API core functions and clean-room `cuda_fp16.h` / `device_launch_parameters.h` headers.

# Third-party notices

**Date:** 2026-09-21  
**Status:** Working inventory. Confirm versions on the buyer’s build host.

## Runtime / build dependencies (not redistributed by this repo)

| Component | Role | Notes |
| --- | --- | --- |
| AMD ROCm / HIP SDK | Device runtime + `hipcc` | Install from AMD; subject to AMD terms |
| hipBLAS | GEMM paths | Optional per CMake flags |
| hipFFT | C2C FFT paths | Optional; disabled build returns deterministic unavailable |
| CMake, MSVC or host C++ toolchain | Native build | Buyer environment |
| Python 3.10+ | `compat` analyzer / bindings | Stdlib-first analyzer; optional inference extras |

## In-tree original work

Project headers under `include/`, `compat/` tooling, kernels authored here, docs, and brand assets created for CUDAtoAMD are part of the repository’s asserted original works (subject to counsel / chain of title).

## Explicitly not bundled

- NVIDIA CUDA Toolkit, `nvcc`, cuDNN, NCCL binaries
- Prebuilt CUDA wheels from PyPI that embed NVIDIA device code
- AMD proprietary blobs beyond what the buyer installs via ROCm

## Attribution

If a future dependency is vendored into this tree, add an SPDX identity and license text here before release. Current design prefers **system-installed** HIP/ROCm rather than vendoring AMD binaries.

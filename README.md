<p align="center">
  <img src="assets/cudatoamd-logo.svg" width="720" alt="CUDAtoAMD â€” CUDA-oriented workflows, native AMD execution">
</p>

<p align="center">
  <a href="https://github.com/theworker02/CUDAtoAMD/releases"><img src="https://img.shields.io/github/v/release/theworker02/CUDAtoAMD?display_name=tag&sort=semver&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/theworker02/CUDAtoAMD" alt="source-available proprietary license"></a>
  <a href="https://github.com/theworker02/CUDAtoAMD"><img src="https://img.shields.io/badge/platform-Windows%20%2B%20ROCm-4866c6" alt="Windows and ROCm"></a>
  <a href="COMPATIBILITY.md"><img src="https://img.shields.io/badge/status-experimental-f4a261" alt="Experimental status"></a>
</p>

<h1 align="center">CUDA-to-AMD Compatibility Runtime</h1>

<p align="center"><strong>Assess CUDA-oriented workloads, compile supported source paths, and execute native AMD code through HIP/ROCmâ€”without pretending an AMD GPU is an NVIDIA GPU.</strong></p>

CUDAtoAMD is an open, clean-room developer runtime for incrementally adapting supported CUDA-oriented source workflows to AMD's HIP/ROCm ecosystem. It offers a CUDA-facing host API subset, an explicit native AMD runtime, bounded source/PTX compilation paths, diagnostic tooling, and conservative wheel admission checks.

> [!WARNING]
> CUDAtoAMD is experimental. It is **not** a universal CUDA replacement and does not run arbitrary CUDA binaries, CUDA-locked Python wheels, or full CUDA frameworks unchanged. Unsupported behavior is reported explicitly rather than silently emulated.

## What is in v1.4.0

- Native HIP device initialization that reports the selected AMD GPU truthfully and can run a numerical GPU self-test.
- Clean-room CUDA Runtime and Driver API compatibility subsets, plus explicit AMD C/C++ APIs for streams, pools, events and HSACO modules.
- A bounded CUDA-syntax and PTX compilation route to AMD code objects through HIP tools.
- FP32/mixed-precision GEMM subsets, optional C2C FFT, selected neural operators, graph capture/replay, and explicit Python bindings.
- CUDA source inventory, toolchain discovery, wheel preflight, release validation and a GitHub Pages project site.

## Why this exists

CUDA-oriented codebases often mix portable host logic with NVIDIA-specific APIs, build tools, libraries and binary assumptions. HIPIFY is valuable for source migration, but it is not a drop-in runtime for every source tree or prebuilt application. CUDAtoAMD focuses on the boundary developers can inspect and verify:

```text
CUDA-oriented source or explicit AMD code object
                    â”‚
                    â–¼
        CUDAtoAMD headers / developer tools
                    â”‚
                    â–¼
      compatibility ABI and semantic checks
                    â”‚
                    â–¼
             HIP / ROCm on AMD hardware
```

The project does not spoof `nvidia-smi`, fabricate an NVIDIA compute capability, redistribute NVIDIA DLLs, or claim that AMD hardware is CUDA hardware. It aims to make the supported route practical and diagnosable for local developers.

## Start here

### 1. Inspect a CUDA-oriented source tree

The analyzer requires Python 3.10+ and has no third-party dependencies. It is read-only: it inventories CUDA headers, APIs, constructs and PTX inputs without changing the target project.

```powershell
python -m compat analyze path/to/cuda-project
python -m compat analyze path/to/cuda-project --format json
python -m compat port-plan path/to/cuda-project
python -m compat port-plan path/to/cuda-project --format json --output port-plan.json
python -m compat matrix
python -m compat doctor
```

Use the analyzer and **port-plan** output to decide whether a workload fits the implemented surface before attempting a port or native build. `port-plan` prioritizes UNSUPPORTED/NVIDIA_SPECIFIC findings first and adds per-file actions and effort hints; it does **not** rewrite sources. See [PORTING.md](PORTING.md).

### 2. Build the native runtime on Windows

The verified host is Windows with Visual Studio Build Tools, ROCm/HIP SDK 7.1 and an AMD RX 7800 XT (`gfx1101`). Start a 64-bit Visual Studio developer prompt.

```bat
:: Backend-free ABI build: no GPU SDK required
cmake -S . -B build-msvc -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=OFF -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-msvc
ctest --test-dir build-msvc --output-on-failure

:: HIP build: HIP and hipBLAS required; hipFFT is optional
set CMAKE_PREFIX_PATH=C:\Program Files\AMD\ROCm\7.1
cmake -S . -B build-hip -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-hip
ctest --test-dir build-hip --output-on-failure
```

If hipFFT is unavailable, use `-DCOMPATCUDA_ENABLE_FFT=OFF`. The runtime returns a deterministic backend-unavailable result for the FFT API rather than loading a missing DLL.

### 3. Check the real AMD execution path

```powershell
python -m compat toolchain --format json
python -m compat init --library build-hip/compatcuda.dll --self-test --format json
python -m compat doctor --native
```

`compat init` loads the named native library, runs `amdInit`, selects the requested HIP device and reports actual name, architecture, memory, compute-unit and wave-size properties. With `--self-test`, it also executes a small native softmax check. It does not manufacture an NVIDIA device report.

## Developer workflows

### Explicit AMD module loading

The native C API accepts already-compiled AMDGPU code objects (`.hsaco`) and exposes explicit module loading, function lookup, stream-bound dispatch, events and memory pools. This is the most deterministic route for native AMD execution.

```c
#include "amd_runtime.h"

AmdDevice device;
AmdStream stream;
AmdMemPool pool;
AmdModule module;
AmdFunction kernel;

amdInit(0);
amdGetDevice(&device, 0);
amdStreamCreate(device, &stream);
amdMemPoolCreate(device, 64 * 1024 * 1024, &pool);
amdModuleLoadFile(device, "vector_add.hsaco", &module);
amdModuleGetFunction(&kernel, module, "vector_add");
/* Allocate, bind arguments, launch, synchronize and destroy explicitly. */
```

See [the native runtime contract](docs/native-runtime.md) and [C++ RAII wrapper](include/amd_runtime.hpp) for the complete ownership model.

### CUDA-facing host subset

The project ships clean-room headers for a documented subset of CUDA Runtime, Driver, cuBLAS and cuFFT-shaped APIs. They are intended for source-oriented experiments and supported host workflows, not for replacement DLL naming or binary interception.

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Supported calls enter the compatibility ABI and dispatch to HIP/hipBLAS.
// Check every result; unsupported calls fail deterministically.
cudaError_t status = cudaDeviceSynchronize();
```

Read [CUDA source compatibility](docs/cuda-source-compatibility.md) before relying on a symbol. The capability database and source analyzer are the authority for the implemented classification, not the presence of a similarly named header.

### CUDA-oriented source and bounded PTX compilation

`compat cc` invokes external HIPIFY/hipcc for the narrow source path it supports. The PTX tools accept a tested subset and fail closed on operations that have not been lowered or tested.

```powershell
python -m compat toolchain --format json
python -m compat cc examples/cuda_vector.cu -o vector_add.hsaco
python -m compat ptx examples/vector_add_bounded.ptx -o vector_add.hsaco
```

Supported compiler features and known exclusions are documented in [PTX subset support](docs/ptx-subset.md). This is not a general NVVM/PTX compiler, cubin translator or CUDA binary JIT.

### Python and framework-adjacent code

The optional bindings provide explicit lifecycle management around the native library. NumPy and CPU-PyTorch paths stage supported FP32 inference operations through the native runtime. Autograd, `torch.cuda`, custom CUDA wheels and framework binary interception are intentionally outside this surface.

```powershell
python -m pip install ".[inference]"
python -m compat demo --operation softmax
python -m compat framework-doctor --format json
```

See [Python native bindings](docs/python-native.md) and [framework integration](docs/framework-integration.md).

### Wheel admission, not wheel spoofing

Use the wheel tools before executing third-party packages that may carry CUDA-native extensions:

```powershell
python -m compat wheel-doctor path/to/package.whl --format json
python -m compat python --wheel path/to/package.whl --wheel-sha256 <reviewed-sha256> -- script.py
```

CUDAtoAMD blocks known CUDA-linked extension patterns unless a workflow has been explicitly reviewed. It does not replace CUDA DLL imports or claim transparent CUDA-wheel execution. Details: [CUDA wheel execution](docs/cuda-wheel-execution.md).

## Implemented capability map

| Area | Included now | Important boundary |
| --- | --- | --- |
| Device/runtime | HIP initialization, device properties, allocation, copies, streams, events, pools | Native HIP availability is required |
| Modules | `.hsaco` file/memory loading, symbol lookup, kernel launch | No cubin or arbitrary CUDA binary translation |
| CUDA-facing APIs | Documented Runtime and Driver subsets | Not a complete CUDA ABI or replacement vendor DLL |
| Math | FP32/mixed-precision GEMM subset; optional C2C FFT | Not cuBLAS/cuFFT parity; hipFFT may be unavailable |
| Neural ops | RMSNorm, softmax, SwiGLU, interleaved RoPE | Correctness-first, not production-tuned kernels |
| Graphs | Single-stream native capture, instantiate, replay, cancellation | CUDA graph facade remains limited |
| Compiler | Bounded CUDA-syntax workflow and tested PTX subset | No general PTX/NVVM frontend or tensor-core translation |
| Python | CFFI lifecycle and CPU-staged NumPy/PyTorch inference | No autograd, `torch.cuda` or transparent wheel support |
| Diagnostics | Source analysis, port-plan, compatibility matrix, toolchain, init and release checks | Results describe known scope, not a compatibility percentage |

## Verification evidence

On the verified Windows host, the HIP build passes six native CTests, including module correctness, CUDA contract and core HIP tests. The HIP-disabled build passes four deterministic ABI/error-path CTests. The Python suite covers analysis, source/PTX boundaries, native lifecycle, module loading, mixed GEMM, FFT, graphs, neural operations, wheel admission, release metadata and the Pages site structure.

The native vector-add fixture has been checked through both file and memory module loading. The current release has also exercised real `compat init --self-test` execution on the RX 7800 XT. This is meaningful narrow correctness evidenceâ€”not a benchmark, production certification or a claim about every AMD GPU, Linux, CUDA application or AI framework.

## Documentation

| Need | Read |
| --- | --- |
| Build and first native run | [Local developer guide](docs/local-developer-guide.md) |
| Linux / ROCm paths | [Linux developer guide](docs/linux-developer-guide.md) |
| Windows packaging & deploy | [Windows deployment](docs/windows-deployment.md) |
| Analyze → prioritize → HIPIFY | [Porting workflow](PORTING.md) (`compat port-plan`) |
| Real HIP initialization | [Initialization guide](docs/initialization.md) |
| CUDA-facing source path | [CUDA source compatibility](docs/cuda-source-compatibility.md) |
| PTX syntax and limits | [PTX subset](docs/ptx-subset.md) |
| C/C++ native ABI | [Native runtime](docs/native-runtime.md) |
| Math and neural APIs | [Native math](docs/native-math.md) |
| Python lifecycle | [Python native bindings](docs/python-native.md) |
| PyTorch-adjacent adapter | [Framework integration](docs/framework-integration.md) |
| AI profile honesty | [AI compatibility](docs/ai-compatibility.md) |
| Drop-in contract | [Drop-in mode](docs/drop-in-mode.md) |
| CUDA wheel safety | [Wheel execution](docs/cuda-wheel-execution.md) |
| Version and release validation | [Release checklist](RELEASING.md) |
| API compatibility inventory | [Compatibility matrix](COMPATIBILITY.md) |
| System design | [Architecture](ARCHITECTURE.md) |
| Acquisition / diligence | [ACQUISITION.md](ACQUISITION.md), [Acquisition guide](docs/ACQUISITION_GUIDE.md), [Data room](docs/acquisition/) |

## Repository layout

```text
assets/          Project identity assets
bindings/        Explicit language bindings
compat/          Python CLI, source analysis, compiler and wheel diagnostics
compatibility/   Capability inventory data
docs/            Developer, API and release documentation
docs/site/       Dependency-free GitHub Pages project site
examples/        CUDA-oriented source and bounded PTX examples
include/         Public C, C++ and CUDA-facing compatibility headers
kernels/         Native HIP neural kernels
runtime/         Internal compatibility ABI contract
src/             Runtime, driver, graph, math and HIP adapter implementation
tests/           Python and native correctness/contract coverage
```

## Installation and distribution

The repository is the primary distribution channel. A Python wheel contains the analyzer and developer tools; it intentionally does **not** bundle HIP/ROCm vendor DLLs or AMD GPU code objects.

```powershell
python -m pip install cuda-amd-compat
python -m compat doctor
```

For an authoritative local package artifact, build from a checked source checkout:

```powershell
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
python -m compat release-check --library build-hip/compatcuda.dll --format json
```

The GitHub Pages source lives in [`docs/site/`](docs/site/). Once Pages is enabled in the repository settings with **GitHub Actions** as the source, the included workflow deploys it from `main`. The final URL is normally `https://theworker02.github.io/CUDAtoAMD/`.

## Security, legal and clean-room boundaries

- The project is source-available proprietary licensed; see [LICENSE](LICENSE).
- It does not redistribute NVIDIA software, drivers or proprietary libraries; see [LEGAL.md](LEGAL.md) and [THIRD_PARTY.md](THIRD_PARTY.md).
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
- Validate untrusted source trees, PTX and wheel inputs before execution. The toolchain deliberately rejects unsupported constructs instead of attempting unsafe implicit conversions.
- Do not treat this runtime as a sandbox or a substitute for GPU-driver security updates.

## Near-term direction

The practical next steps are deeper testing and incremental, documented expansion of the supported surface: additional source/compiler constructs, more numerical conformance coverage, broader AMD architecture validation, and native math library adapters where their ROCm dependencies are actually available. Full CUDA API, PTX/NVVM and transparent CUDA-framework parity are not represented as current milestones.

## Contributing

Contributions should preserve the projectâ€™s clean-room and truthful-compatibility principles. Please include focused tests, update the capability documentation, distinguish verified behavior from planned work, and avoid adding vendor binaries, credentials or opaque generated artifacts to the repository.

## License

**Source-available proprietary** — evaluation under [LICENSE](./LICENSE); commercial / production use via [COMMERCIAL.md](./COMMERCIAL.md). See [LICENSE_TRANSITION_NOTICE.md](./LICENSE_TRANSITION_NOTICE.md) and [NOTICE](./NOTICE).


# Windows Deployment Guide

**Audience:** developers packaging or validating CUDAtoAMD on Windows with AMD
ROCm/HIP.  
**Verified reference host (project docs):** Windows + Visual Studio Build Tools +
ROCm/HIP SDK 7.1 + AMD RX 7800 XT (`gfx1101`). Other GPUs and SDK versions are
unverified until you run the same tests locally.

This guide describes what Windows deployment **is** and **is not**. It does not
claim that arbitrary CUDA applications or CUDA Python wheels run unchanged.

---

## Deployment contract

| Mode | What ships | What happens at runtime |
| --- | --- | --- |
| ABI / smoke build (`COMPATCUDA_ENABLE_HIP=OFF`) | `compatcuda.dll` + headers/import lib | State-changing calls return `cudaErrorNotSupported`; device count is zero |
| HIP-enabled build (`COMPATCUDA_ENABLE_HIP=ON`) | Same ABI surface backed by HIP/hipBLAS (FFT optional) | Supported subset dispatches to HIP; unsupported calls still fail explicitly |
| Explicit native AMD path | C/C++ `amd_*` APIs, Python `compat.native` / `compat.neural` | Truthful AMD device reports; HSACO module load/launch |

Compatibility deployment must be **explicit**:

- Build applications against the project headers and import library, **or**
- Launch with an owner-configured environment (`compat run`, `compat python`,
  `AMD_RUNTIME_LIB_PATH`).

The project will **not** use DLL injection, stealth interposition, renamed
NVIDIA DLLs, or `nvidia-smi` spoofing.

---

## Prerequisites

1. **64-bit Visual Studio** (or Build Tools) with C++ workload — use an
   “x64 Native Tools” developer prompt for CMake/`nmake`.
2. **AMD GPU driver** appropriate for your card (install separately).
3. **AMD ROCm/HIP SDK for Windows** (reference docs use 7.1 under
   `C:\Program Files\AMD\ROCm\7.1`).
4. **Python 3.10+** for the `compat` CLI and optional bindings.
5. Optional: **hipBLAS** (required for GEMM paths), **hipFFT** (optional; disable
   with `-DCOMPATCUDA_ENABLE_FFT=OFF` if absent).

Discover the toolchain:

```powershell
python -m compat doctor
python -m compat toolchain --format json
```

On Windows, `compat doctor` resolves `hipcc` from `PATH` or from
`%ProgramFiles%\AMD\ROCm\*\bin\hipcc.exe` (newest version directory first).

---

## Build recipes

### Backend-free ABI smoke build

Useful for headers, link surface, and fail-closed error paths without a GPU SDK:

```bat
cmake -S . -B build-msvc -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=OFF -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-msvc
ctest --test-dir build-msvc --output-on-failure
```

Expect deterministic `cudaErrorNotSupported` behavior for state-changing CUDA-facing
calls. Do not ship this build as a “GPU runtime.”

### HIP-enabled functional build

```bat
set CMAKE_PREFIX_PATH=C:\Program Files\AMD\ROCm\7.1
cmake -S . -B build-hip -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-hip
ctest --test-dir build-hip --output-on-failure
```

If hipFFT is missing:

```bat
cmake -S . -B build-hip -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_ENABLE_FFT=OFF -DCOMPATCUDA_BUILD_TESTS=ON
```

The FFT API then returns a deterministic backend-unavailable result rather than
loading a missing DLL.

### Neural code object

Neural demos and `compat init --self-test` need `neural.hsaco` built for your
`gfx` target (see CMake `COMPATCUDA_TEST_ARCH` / neural target). Mismatched
architecture yields “no compatible kernel image,” not silent fallback.

---

## Runtime configuration

| Variable / flag | Purpose |
| --- | --- |
| `AMD_RUNTIME_LIB_PATH` | Absolute path to the built `compatcuda.dll` for Python/native loaders |
| `AMD_RUNTIME_ROCM_ROOT` | Optional: pin the ROCm root when multiple SDKs exist |
| `python -m compat init --library …` | Load library, `amdInit`, report real device props |
| `python -m compat run PROGRAM` | Prepend discovered ROCm `bin` to `PATH`; **no** DLL injection |
| `python -m compat python SCRIPT` | Same PATH policy for a Python child process |

Example:

```powershell
$env:AMD_RUNTIME_LIB_PATH = (Resolve-Path build-hip\compatcuda.dll).Path
python -m compat init --library build-hip\compatcuda.dll --self-test --format json
python -m compat doctor --native
python -m compat demo --operation softmax
```

---

## Packaging and install layout

CMake install rules stage the runtime library, public headers, license, and
related notices (see `RELEASING.md`). Consumer builds typically leave tests off
unless `-DCOMPATCUDA_BUILD_TESTS=ON`.

Python packaging (`cuda-amd-compat` / editable install) ships the analyzer and
CLI. It **does not** bundle HIP/ROCm vendor DLLs or AMDGPU code objects — those
come from the local SDK and your HIP build.

Release gate example:

```powershell
python -m compat release-check --library build-hip\compatcuda.dll --format json
```

---

## Application integration patterns (supported)

1. **Link against headers + import library** for the documented CUDA-facing
   subset; check every return code.
2. **Prefer the explicit AMD C/C++ API** (`amd_runtime.h` / `amd_runtime.hpp`)
   when you control the call sites — clearest ownership model.
3. **Load HSACO modules** you compiled yourself (`compat code-object`,
   `compat ptx-object`, or hipcc).
4. **Python Session / AmdInferenceModule** for CPU-staged FP32 inference ops
   (softmax, rmsnorm, swiglu, rope) — not training, not `torch.cuda`.

---

## Anti-patterns (unsupported / refused)

- Renaming `compatcuda.dll` to a CUDA vendor DLL name and hoping apps load it.
- Injecting the DLL into third-party processes.
- Shipping CUDA `.cubin` / `.fatbin` and expecting translation.
- Claiming Windows multi-GPU collectives readiness (`compat profile ai` reports
  collectives as `WINDOWS_BLOCKED` where applicable).
- Treating wheel preflight “unverified” as “safe to run as CUDA.”

---

## Validation checklist before calling a build “deployable”

1. `ctest` passes for the configuration you ship (ABI and/or HIP).
2. `compat doctor` / `compat toolchain` show the intended `hipcc`.
3. `compat init --self-test` succeeds on the target GPU when neural ops matter.
4. `compat release-check` is green for the staged library and docs set.
5. You recorded OS, driver, ROCm, `gfx` arch, and test commands in your release notes.

---

## Related documentation

- [Local developer guide](local-developer-guide.md) — day-one Windows workflow
- [Initialization](initialization.md) — `compat init` contract
- [Drop-in mode](drop-in-mode.md) — what `DROP_IN` means today
- [Linux developer guide](linux-developer-guide.md) — Unix/ROCm paths
- [PORTING.md](../PORTING.md) — analyze / port-plan / HIPIFY loop
- [RELEASING.md](../RELEASING.md) — clean build and metadata gates

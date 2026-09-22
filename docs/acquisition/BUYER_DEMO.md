# Buyer Demo — CUDAtoAMD

**Date:** 2026-09-22  
**Target:** fresh machine → clone → install → analyze → tests → (optional GPU)  
**Contact:** [@theworker02](https://github.com/theworker02)

This script supports **technical diligence**, not production certification. Minimal path requires **Python 3.10+** only. GPU/HIP steps need AMD ROCm/HIP installed and suitable hardware.

---

## Phase A — CPU / tooling (required)

Works without a GPU or ROCm installation.

### Linux / macOS shell

```bash
git clone https://github.com/theworker02/CUDAtoAMD.git
cd CUDAtoAMD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m compat doctor
python -m compat analyze examples
python -m compat port-plan examples
python -m compat matrix | head -40
python -m compat toolchain --format json
python -m unittest discover -s tests -v
```

### Windows PowerShell

```powershell
git clone https://github.com/theworker02/CUDAtoAMD.git
cd CUDAtoAMD
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python -m compat doctor
python -m compat analyze examples
python -m compat port-plan examples
python -m compat matrix
python -m compat toolchain --format json
python -m unittest discover -s tests -v
```

### Expected results (Phase A)

| Step | Pass criteria |
| --- | --- |
| `doctor` | Prints host info; reports hipcc found or not found; exits without crash |
| `analyze` / `port-plan` | Exit 0; inventory or structured plan on `examples/` |
| `matrix` | Renders compatibility table from in-repo database |
| `toolchain` | JSON listing discovered ROCm/HIP paths or explicit absence |
| `unittest` | Analyzer, port-plan, database, PTX/compiler boundaries, wheel/release/site tests pass **without GPU** |

Failures in Phase A should be investigated before blaming missing GPU hardware.

---

## Phase B — Optional native build (HIP)

Requires Visual Studio Build Tools (Windows) or equivalent toolchain, ROCm/HIP SDK, and AMD GPU for execution tests.

Reference Windows host (from README): ROCm/HIP SDK 7.1, RX 7800 XT (`gfx1101`).

```bat
:: From x64 Visual Studio developer prompt (Windows example)
set CMAKE_PREFIX_PATH=C:\Program Files\AMD\ROCm\7.1
cmake -S . -B build-hip -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-hip
ctest --test-dir build-hip --output-on-failure
```

HIP-disabled ABI smoke (no GPU SDK required):

```bat
cmake -S . -B build-msvc -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=OFF -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-msvc
ctest --test-dir build-msvc --output-on-failure
```

Linux: see [../linux-developer-guide.md](../linux-developer-guide.md).  
Windows details: [../local-developer-guide.md](../local-developer-guide.md), [../windows-deployment.md](../windows-deployment.md).

### Expected results (Phase B)

| Step | Pass criteria |
| --- | --- |
| HIP-disabled CTest | Documented ABI/error-path tests pass (README cites four on reference host) |
| HIP-enabled CTest | Documented native tests pass when ROCm stack matches (README cites six on reference host) |
| Missing hipFFT | Build with `-DCOMPATCUDA_ENABLE_FFT=OFF` if needed; FFT API returns deterministic unavailable |

---

## Phase C — Optional runtime demo (GPU)

After Phase B produces `compatcuda` (path varies by platform):

```powershell
python -m pip install -e ".[inference]"
python -m compat init --library build-hip/compatcuda.dll --self-test --format json
python -m compat demo --operation softmax --library build-hip/compatcuda.dll
```

Adjust library path extension (`.so` on Linux) and build directory to match your build.

### Expected results (Phase C)

| Step | Pass criteria |
| --- | --- |
| `init` | Reports **real** AMD device name and architecture — not an NVIDIA spoof |
| `init --self-test` | Numerical self-test passes on supported configuration |
| `demo` | GPU softmax matches NumPy reference; mismatch is a **fail**, not ignored |

Absence of GPU or ROCm is **not** a pass for Phase C — skip or document as N/A.

---

## Phase D — Optional diligence extras

```powershell
python -m compat wheel-doctor path/to/sample.whl --format json
python -m compat framework-doctor --format json
python -m compat release-check --library build-hip/compatcuda.dll --format json
```

Use only wheels and sources you are authorized to analyze. Wheel tools perform **admission/diagnostics**, not CUDA extension translation.

Bounded compile examples (requires toolchain from Phase A/B):

```powershell
python -m compat cc examples/cuda_vector.cu -o vector_add.hsaco
python -m compat ptx examples/vector_add_bounded.ptx -o vector_add.hsaco
```

Unsupported PTX or CUDA constructs should **fail closed** — that behavior is correct, not a demo defect.

---

## Out of scope for this demo

- Transparent CUDA wheel execution or CUDA DLL replacement
- Full PyTorch CUDA training or `torch.cuda` drop-in
- Multi-GPU / NCCL production readiness claims
- Performance benchmarks or SLAs not checked into this repository
- Legal title, valuation, or exclusivity adjudication

---

## See also

- [KNOWN_LIMITATIONS.md](./KNOWN_LIMITATIONS.md)
- [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)
- [../ACQUISITION_GUIDE.md](../ACQUISITION_GUIDE.md)
- [../../ACQUISITION.md](../../ACQUISITION.md)
- [../../PORTING.md](../../PORTING.md)
- [../../COMPATIBILITY.md](../../COMPATIBILITY.md)

No valuation is stated in this document.

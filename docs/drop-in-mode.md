# Drop-In Mode: Current Contract

`DROP_IN` is a **planned execution profile name** and a fail-closed ABI posture —
not a claim that arbitrary CUDA applications run today. This page is the
authoritative prose contract for what “drop-in” means in CUDAtoAMD.

---

## Summary in one paragraph

The repository ships an independently implemented **declaration subset** under
`include/` and a `compatcuda` shared library that exports CUDA-shaped symbols
for a documented inventory. Without a HIP backend, every state-changing
operation fails with `cudaErrorNotSupported`, output pointers are nulled, and
device count is zero. With HIP enabled, a **partial** subset dispatches to AMD
via HIP/hipBLAS/optional hipFFT. Unsupported calls still fail explicitly.
Headers alone do **not** turn MSVC or gcc into a CUDA-language compiler.

---

## What is shipped for the declaration subset

- Clean-room headers approximating selected CUDA Runtime / Driver / math API
  shapes used by this project’s tests and examples
- `compatcuda` shared library (`.dll` on Windows, `.so` on Unix when built)
- Compatibility database (`compatibility/cuda_api.json`) classifying symbols as
  `DIRECT`, `ADAPTED`, `PARTIAL`, `UNSUPPORTED`, `NVIDIA_SPECIFIC`, etc.
- Analyzer and `compat port-plan` for source trees that *call* these APIs

What is **not** shipped as drop-in:

- A renamed `nvcuda.dll` / `libcuda.so` substitute for binary interception
- Automatic loading into third-party processes
- Full CUDA Driver/Runtime surface
- Cubin/fatbin execution

---

## Fail-closed backend-free behavior

Build with `COMPATCUDA_ENABLE_HIP=OFF` (default for ABI smoke):

| Observation | Expected |
| --- | --- |
| `cudaGetDeviceCount` | Reports zero devices (or equivalent fail-closed policy in tests) |
| Allocations / launches / state changes | `cudaErrorNotSupported` (or documented NOT_SUPPORTED) |
| Accidental “success” on GPU work | Must not occur |

This makes mis-deployment **visible**. Applications that ignore error codes can
still be wrong — callers must check status.

---

## HIP-backed subset behavior

When `COMPATCUDA_ENABLE_HIP=ON` and the ROCm/HIP stack is present:

- Supported device, memory, stream, and event operations may dispatch through
  an internal HIP adapter
- Logical allocation registry and logical streams/events keep **HIP handles out
  of the public CUDA-facing ABI**
- Device properties report **AMD** capabilities without NVIDIA impersonation
- Library adapters (e.g. GEMM) follow the inventory; unsupported combinations
  return NOT_SUPPORTED

Always validate with CTest and `compat init` on the target machine. Inventory
status `DIRECT` means “documented analogue,” not “proven for your app.”

---

## Language / compiler boundary (critical)

CUDA launch syntax (`<<<>>>`), `__global__` device code, and related language
extensions require **hipcc / Clang CUDA-language mode** or an explicit rewrite
to HIP launch APIs / HSACO modules.

| Approach | Drop-in? |
| --- | --- |
| Link host code against headers + `compatcuda` | Only for the implemented host subset |
| Compile `.cu` with MSVC/gcc as if it were CUDA | **No** |
| `compat cc` (HIPIFY → hipcc) | Explicit source pipeline, not silent drop-in |
| `compat code-object` / `compat ptx-object` | Bounded AOT paths to AMD ELF/HSACO |
| Load existing NVIDIA cubin | **No** |

---

## Explicit alternatives preferred over “drop-in”

For new work, prefer:

1. **Native AMD API** — `amd_runtime.h` / C++ RAII wrapper
2. **Explicit Python Session** — NumPy/CPU-PyTorch staging
3. **Port-plan → HIPIFY tree** — see [PORTING.md](../PORTING.md)

These paths make ownership, error handling, and architecture targeting obvious.

---

## Launch helpers vs injection

Commands such as `compat run` and `compat python`:

- Prepend the discovered ROCm SDK `bin` directory to `PATH`
- Optionally set `AMD_RUNTIME_LIB_PATH`
- Do **not** inject DLLs, rewrite imports, or claim an AMD GPU is NVIDIA hardware

That is “explicit environment,” not classic drop-in interception.

---

## Checklist before using DROP_IN language externally

1. State which build flags and SDK versions were tested.
2. List the exact APIs exercised (or point at CTest names).
3. Disclose unsupported families (collectives, full DNN, RAND/SPARSE/SOLVER, etc.).
4. Never present wheel admission tools as proof of drop-in wheel execution.
5. Point readers to [COMPATIBILITY.md](../COMPATIBILITY.md) as an inventory, not
   a success percentage.

---

## Related documentation

- [Windows deployment](windows-deployment.md)
- [CUDA source compatibility](cuda-source-compatibility.md)
- [CUDA wheel execution](cuda-wheel-execution.md)
- [AI compatibility](ai-compatibility.md)
- [Architecture](../ARCHITECTURE.md)

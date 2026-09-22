# AI Compatibility Profile

**Machine-readable companion:** `spec/ai-compatibility.yaml` (roadmap/status
manifest).  
**CLI:** `python -m compat profile ai`

This document explains what the AI-oriented profile **reports**, what the
runtime **actually implements**, and what buyers or developers must **not**
infer. It is not a training-readiness certification and not a CUDA/cuDNN parity
statement.

---

## How to read the profile

```powershell
python -m compat profile ai
python -m compat framework-doctor --format json
python -m compat doctor --cuda-api
```

The profile text summarizes layers:

| Layer | Typical statuses in this project |
| --- | --- |
| Runtime (device, alloc, copy, stream, event) | Implemented when built with `COMPATCUDA_ENABLE_HIP=ON` and validated |
| Math (GEMM / FFT) | FP32 and mixed-precision GEMM subset via hipBLAS; optional complex 1D FFT via hipFFT |
| Neural ops | FP32 RMSNorm, softmax, SwiGLU, interleaved RoPE via native HSACO (explicit session) |
| Framework | Explicit CPU float32 PyTorch bridge (`AmdInferenceModule`); **no** autograd / `torch.cuda` |
| Distributed | Collectives called out as blocked or unimplemented on Windows paths |
| Training | **SINGLE GPU: NOT READY** / **MULTI GPU: NOT READY** (printed by the profile) |

If `hipcc` is missing, the profile reports the HIP compiler as `UNAVAILABLE`.
That is an environment fact, not a claim that installing hipcc unlocks full AI
frameworks.

---

## What is implemented (honest inventory)

### Runtime core (HIP build)

- Device init and truthful AMD property reporting (`compat init`, native `amdInit`)
- Allocation, memcpy/memset-style paths in the documented subset
- Streams and events with logical wrappers (not raw HIP handles in the public
  CUDA-facing ABI)
- Optional graph capture/replay on the **native** AMD path (single-stream;
  see local developer guide for lifetime rules)

### Math adapters

- FP32 / FP16 / BF16 **input** GEMM with FP32 output path (hipBLAS-backed;
  unsupported type combos return NOT_SUPPORTED rather than silent cast)
- Optional complex FP32 1D FFT when hipFFT is enabled at build time
- BLASLt, RAND, SPARSE, SOLVER: **not implemented** (profile says so)

### Neural operators

- Correctness-first FP32 kernels: RMSNorm, softmax, SwiGLU, RoPE
- Require a matching `neural.hsaco` for the GPU architecture
- Useful for local experiments and numerical self-tests — not FlashAttention,
  convolution stacks, or quantized production kernels

### Framework-adjacent path

- `compat.framework.AmdInferenceModule` wraps the native session as a closeable
  `torch.nn.Module` when PyTorch is installed
- CPU contiguous finite float32 tensors only; `requires_grad=False`
- Explicit host→device→host staging; transfer overhead is expected
- Does **not** install a PyTorch device backend or intercept CUDA extensions

See [framework-integration.md](framework-integration.md) for the API contract.

---

## What the YAML file is for

`spec/ai-compatibility.yaml` is a structured roadmap/status list. Individual
entries may lag or lead prose docs; when they disagree with `compat profile ai`,
`COMPATIBILITY.md`, or CTest evidence, **prefer executable tools and tests**.

Treat YAML statuses such as `EXPERIMENTAL`, `PARTIAL`, `UNSUPPORTED`, and
`WINDOWS_BLOCKED` as diligence signals, not marketing tiers.

---

## Training and multi-GPU readiness

The CLI profile intentionally prints:

```text
Training readiness
  SINGLE GPU: NOT READY
  MULTI GPU: NOT READY
```

Reasons include (non-exhaustive):

- No autograd / optimizer integration
- No transparent framework device
- Collectives / RCCL path not productized here (Windows called out as blocked)
- No claim of cuDNN / full MIOpen surface
- Neural kernels are a tiny FP32 subset

Do not override this in slides or acquisition materials without new,
documented conformance evidence.

---

## Recommended evaluation workflow for AI-ish workloads

1. Run `compat analyze` / `compat port-plan` on the CUDA or framework-adjacent
   source tree — inventory NCCL, cuDNN, custom CUDA extensions, and PTX.
2. Run `compat wheel-doctor` on any third-party wheels — expect `blocked` for
   CUDA-linked native extensions.
3. Build HIP runtime; run `compat init --self-test` and `compat demo`.
4. If PyTorch CPU-staged ops are enough, try `AmdInferenceModule` for the four
   supported ops only.
5. For everything else, plan a **native HIP/ROCm** or framework-ROCm rebuild —
   not “drop in the CUDA wheel.”

---

## Windows vs Linux notes

- Windows is the historically verified host for many native tests in this repo.
- Linux ROCm installs are supported for toolchain discovery
  (`/opt/rocm`, `$ROCM_PATH`) via `compat doctor`; full parity of every Windows
  CTest on every Linux distro is **not** asserted here.
- See [linux-developer-guide.md](linux-developer-guide.md) and
  [windows-deployment.md](windows-deployment.md).

---

## Related commands and docs

| Resource | Use |
| --- | --- |
| `compat profile ai` | Print the AI environment summary |
| `compat framework-doctor` | Exact framework bridge capabilities |
| `COMPATIBILITY.md` | Interface inventory (not a success %) |
| [native-math.md](native-math.md) | GEMM/FFT boundaries |
| [cuda-wheel-execution.md](cuda-wheel-execution.md) | Why CUDA wheels are not transparently run |
| [ACQUISITION_GUIDE.md](ACQUISITION_GUIDE.md) | Buyer-oriented technical overview |

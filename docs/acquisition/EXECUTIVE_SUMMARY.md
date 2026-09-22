# Executive Summary — CUDAtoAMD

**Date:** 2026-09-22  
**Current license:** source-available proprietary (see root `LICENSE`)  
**Package version (in-tree):** 1.4.0 (`cuda-amd-compat`)  
**Commercial inquiries:** [`COMMERCIAL.md`](../../COMMERCIAL.md)  
**Acquisition brief:** [`ACQUISITION.md`](../../ACQUISITION.md)

---

## What this is

An experimental **clean-room CUDA-oriented compatibility and native AMD runtime** project. It helps teams:

- **Inventory** CUDA headers, APIs, constructs, and PTX inputs in a source tree (read-only).
- **Prioritize** porting work via `compat port-plan` (UNSUPPORTED / NVIDIA_SPECIFIC first).
- **Compile bounded** CUDA-syntax and validated PTX subsets to AMD code objects (`.hsaco`).
- **Execute explicit HIP/ROCm workloads** through native APIs and a documented CUDA-facing subset.

It does **not** spoof NVIDIA hardware, redistribute NVIDIA binaries, or claim universal CUDA drop-in for binaries, wheels, or frameworks.

---

## Problem addressed

CUDA codebases mix portable logic with NVIDIA-specific APIs, binaries, wheels, PTX/NVVM, and framework assumptions (`torch.cuda`, cuDNN-shaped calls, NCCL, etc.). Teams need **diagnosable AMD pathways** and **honest failure modes** more than silent emulation or marketing parity percentages.

CUDAtoAMD focuses on **inspectable boundaries**: analyzer separate from runtime, compatibility database as inventory authority, fail-closed unsupported paths.

---

## Core asset classes (what diligence is buying)

| Asset | Description |
| --- | --- |
| `compat` CLI / Python package | analyze, port-plan, doctor, matrix, toolchain, cc, ptx, wheel-doctor, init, demo, release-check |
| Native `compatcuda` + headers | CUDA Runtime/Driver subset, explicit AMD APIs, HIP adapter (when enabled) |
| `compatibility/cuda_api.json` | Machine-readable inventory; renders `COMPATIBILITY.md` |
| Documentation + data room | Developer guides, architecture, acquisition materials, GitHub Pages site source |
| Tests | Python unittest modules; native CTests (HIP and HIP-disabled paths) |

---

## Maturity and verification posture

- Positioned as an **experimental subset** (README badge and warnings).
- **Verified narratives** emphasize Windows + ROCm/HIP SDK 7.1 + AMD RX 7800 XT (`gfx1101`) reference host.
- README documents **six** HIP-enabled native CTests and **four** HIP-disabled ABI/error CTests on that path, plus broad Python unittest coverage areas.
- Linux ROCm discovery is implemented for tooling; **not** every test is asserted on every Linux distribution.

Do not infer production readiness for arbitrary CUDA applications from this summary alone.

---

## Capability snapshot (honest)

**Present to varying documented/tested degrees:**

- Read-only analysis and prioritized port plans
- HIP toolchain discovery (Windows and Linux ROCm layouts)
- HIP-backed runtime subset when built with `COMPATCUDA_ENABLE_HIP=ON`
- Bounded CUDA-syntax / PTX → `.hsaco` paths
- FP32/mixed GEMM subset; optional FFT; small FP32 neural op set; limited graph capture/replay
- Python/C bindings; CPU-staged PyTorch bridge for four documented ops
- Wheel **admission** and diagnostics (not transparent CUDA wheel execution)

**Absent / not claimed:**

- Full CUDA API or Driver parity
- Automatic migration or silent rewrite of repositories
- Transparent CUDA binary or wheel execution
- `torch.cuda`, training, or multi-GPU / NCCL readiness (AI training profile: **NOT READY**)
- Fabricated benchmarks or market metrics

Current compatibility matrix tracks **30** interfaces; status breakdown is in root `COMPATIBILITY.md`. **`DIRECT` ≠ proven ABI drop-in.**

---

## Deployment model

- **pip** install or editable install of analyzer/tools (`cuda-amd-compat`); wheel does not bundle ROCm DLLs.
- Optional **CMake** build of native library with or without HIP.
- Explicit environment launch (`compat run`, `compat python`) — **no DLL injection**.

---

## Language / stack

- Python ≥ 3.10 (setuptools); analyzer core has no third-party dependencies
- Optional: `cffi`, `numpy`, user-provided PyTorch for inference extras
- C/C++ runtime, CMake, CTest
- External: AMD ROCm/HIP, hipBLAS; hipFFT optional

---

## Licensing posture (factual)

- Current tree: proprietary / source-available terms in root `LICENSE`
- Production / redistribution / SaaS: commercial license via `COMMERCIAL.md`
- Historical transition: `LICENSE_TRANSITION_NOTICE.md`
- **REQUIRES_LEGAL_REVIEW** before treating ownership, exclusivity, or trademark as adjudicated

---

## Ownership (asserted, not adjudicated)

Asserted holder/contact: **theworker02** (https://github.com/theworker02).  
No CLA/DCO program is described in-repo as of this date. Maintenance concentration is a diligence topic.

---

## What a buyer can expect from materials in-repo

- Ability to evaluate analyzer, port-plan, documentation, and (on suitable hardware) native demos
- Clear documentation of **non-goals** (wheels, training, full CUDA, NVIDIA spoofing)
- Diligence packaging that **refuses fabricated metrics**
- Roadmap intent in `ROADMAP.md` (not a contractual commitment)

---

## Top diligence risks (executive view)

1. **Scope gap vs “CUDA on AMD” headlines** — binary/wheel expectations vs actual subset.
2. **ROCm/HIP/driver dependency** — version-sensitive external stack.
3. **License transition** — historical OSS copies vs current proprietary grants.
4. **Single-maintainer concentration** — continuity and bus factor.
5. **Platform skew** — Windows reference host vs buyer’s target Linux fleet.

---

## Next steps

1. Read full [ACQUISITION.md](../../ACQUISITION.md).  
2. Run [BUYER_DEMO.md](./BUYER_DEMO.md).  
3. Contact [@theworker02](https://github.com/theworker02) for commercial or acquisition discussions.

No valuation is stated in this document.

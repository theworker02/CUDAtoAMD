# Known Limitations — CUDAtoAMD

**Date:** 2026-09-22  
**Audience:** Technical and legal diligence  
**Companion:** [ACQUISITION.md](../../ACQUISITION.md), [COMPATIBILITY.md](../../COMPATIBILITY.md)

This document states **known product boundaries and non-claims**. It does not enumerate every unsupported CUDA symbol in existence — see the compatibility inventory for tracked interfaces.

---

## Product / technical limitations

### Not full CUDA

- Runtime, Driver, math, collective, DNN, RAND, SPARSE, and SOLVER coverage is a **documented subset**.
- Root `COMPATIBILITY.md` is generated from `compatibility/cuda_api.json` and is an **interface inventory**, not an application-success rate or parity percentage.
- As of the current matrix, **30** interfaces are tracked with statuses including DIRECT, ADAPTED, PARTIAL, UNSUPPORTED, and NVIDIA_SPECIFIC.
- **`DIRECT` in the database means a documented HIP/ROCm analogue exists in inventory — not that ABI-level CUDA drop-in execution is proven.**

### Not automatic migration

- `compat analyze` is read-only; it does not modify target projects.
- `compat port-plan` prioritizes findings (UNSUPPORTED / NVIDIA_SPECIFIC first) and suggests per-file actions; it **does not rewrite** source trees or guarantee HIPIFY success.
- See [PORTING.md](../../PORTING.md) for the intended human-in-the-loop workflow.

### Not transparent wheels or binaries

- `wheel-doctor` and reviewed `compat python` flows are **admission and environment helpers**.
- Known CUDA-linked extension patterns are **blocked**, not translated at load time.
- No cubin/fatbin translator or arbitrary CUDA binary JIT is offered.
- Details: [../cuda-wheel-execution.md](../cuda-wheel-execution.md).

### Not `torch.cuda` or production training

- Framework integration is an explicit **CPU float32 bridge** for a small set of neural ops (see [../framework-integration.md](../framework-integration.md)).
- AI compatibility profile documents training as **NOT READY** for single- and multi-GPU training (`compat profile ai` narrative in-repo).
- Autograd, custom CUDA wheels, and framework binary interception are outside the documented surface.

### PTX / NVVM / compiler bounds

- Only a **validated PTX subset** is lowered ([../ptx-subset.md](../ptx-subset.md)).
- General PTX, untested NVVM operations, tensor-core translation, and broad framework-generated PTX are out of scope.
- `compat cc` invokes external HIPIFY/hipcc for a **narrow** CUDA-syntax path; failures on unsupported input are expected.

### Math and libraries

- GEMM coverage is FP32/mixed-precision **subset**, adapted via hipBLAS where documented — not cuBLAS parity.
- FFT is optional C2C subset via hipFFT when available; if hipFFT is missing, APIs return deterministic backend-unavailable results rather than loading missing DLLs.
- Neural kernels (RMSNorm, softmax, SwiGLU, interleaved RoPE) are correctness-first, not production-tuned for every AMD SKU.

### Graphs and collectives

- Native single-stream graph capture/replay exists in documented scope; CUDA graph **facade** remains limited.
- NCCL / RCCL entries in inventory may be PARTIAL — not productized multi-GPU training parity.

### DROP_IN profile

- Documented as fail-closed ABI posture / planned profile naming — **not** arbitrary application execution without validation.
- See [../drop-in-mode.md](../drop-in-mode.md).

### Platform and hardware evidence skew

- Primary **verified** narratives in README emphasize Windows + ROCm/HIP SDK 7.1 + RX 7800 XT (`gfx1101`).
- Linux ROCm path discovery is implemented for tooling; **universal** Linux certification on every distro/GPU is **not** claimed.
- Behavior depends on external driver and ROCm versions.

### Security

- The runtime is **not** a sandbox substitute for GPU driver updates or organizational security policy.
- Untrusted source, PTX, and wheels should be validated before execution; unsupported constructs are rejected rather than silently converted.

---

## Licensing / IP limitations

- Current distributions: **source-available proprietary** under root `LICENSE`.
- Production, redistribution, SaaS, and typical commercial use require a **paid commercial license** ([COMMERCIAL.md](../../COMMERCIAL.md)).
- Historical public releases under prior open-source or source-available terms: see [LICENSE_TRANSITION_NOTICE.md](../../LICENSE_TRANSITION_NOTICE.md). Editing the current LICENSE **does not** claw back grants attached to lawful historical copies.
- Asserted copyright contact: **@theworker02**; chain of title, contributor rights, and trademark status **require legal review**.
- No patent portfolio or registered trademark claims appear in this data room.
- No fabricated exclusivity or adjudicated ownership claims.

---

## Third-party dependencies (external to transaction)

| Dependency | Role | Buyer note |
| --- | --- | --- |
| AMD ROCm / HIP | GPU runtime | Version-sensitive; not bundled in Python wheel |
| hipBLAS | GEMM adapter | Required for HIP math path as documented |
| hipFFT | Optional FFT | May be absent; deterministic failure modes |
| HIPIFY / hipcc | Source compilation | Invoked explicitly; AMD toolchain |
| NumPy / PyTorch | Optional Python demos | User-installed; not a CUDA replacement |

See [THIRD_PARTY.md](../../THIRD_PARTY.md) and [LEGAL.md](../../LEGAL.md).

---

## Metrics and affiliation — do not assume

The following are **not claimed** in this data room or acquisition brief:

- GitHub stars, download counts, active users, revenue, or design-win counts
- Endorsement by AMD, NVIDIA, cloud providers, or AI framework vendors
- That optional extras (inference profile, demos) are production-hardened for all SKUs
- That passing Phase A of [BUYER_DEMO.md](./BUYER_DEMO.md) implies GPU production readiness
- That inventory rows imply runtime conformance without matching tests

---

## Related diligence documents

| Document | Why read |
| --- | --- |
| [../../ACQUISITION.md](../../ACQUISITION.md) | Full buyer brief |
| [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) | Short overview |
| [BUYER_DEMO.md](./BUYER_DEMO.md) | Reproducible evaluation |
| [../ai-compatibility.md](../ai-compatibility.md) | AI profile honesty |
| [../../ROADMAP.md](../../ROADMAP.md) | NON-GOAL items |

No valuation is stated in this document.

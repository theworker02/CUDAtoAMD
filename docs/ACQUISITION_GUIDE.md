# CUDAtoAMD — Acquisition Guide (Technical)

**Date:** 2026-09-21  
**Status:** Buyer-oriented technical overview. **No acquisition has occurred** by
virtue of this file.  
**Contact:** GitHub [@theworker02](https://github.com/theworker02) ·
https://github.com/theworker02/CUDAtoAMD

This guide complements root [`ACQUISITION.md`](../ACQUISITION.md) and the data
room under [`acquisition/`](acquisition/). It is factual and intentionally
conservative. It does **not** state valuation, users, revenue, or CUDA parity
percentages.

---

## One-sentence product definition

CUDAtoAMD is an **experimental, clean-room compatibility and developer runtime**
that helps teams **inventory** CUDA-oriented code, **compile bounded** source/PTX
subsets to AMD code objects, and **execute native AMD** work through HIP/ROCm —
without claiming an AMD GPU is an NVIDIA GPU.

---

## Problem it addresses

Organizations with CUDA-oriented codebases need a **diagnosable** path toward AMD
HIP/ROCm: what APIs are used, what is NVIDIA-specific, what can be mapped, and
what must be rewritten. Generic “just HIPIFY everything” advice fails when
binaries, wheels, PTX, and framework lock-in dominate. CUDAtoAMD focuses on
**inspectable boundaries** and **fail-closed** behavior for unsupported work.

---

## What a technical buyer is evaluating

| Asset | Nature |
| --- | --- |
| Python package `cuda-amd-compat` / `compat` CLI | Analyzer, port-plan, doctor, wheel preflight, compilers wrappers, demos |
| Native `compatcuda` library + headers | Documented CUDA-facing subset + explicit AMD APIs |
| Compatibility database | Interface inventory (`DIRECT` ≠ proven drop-in) |
| Docs + acquisition data room | Diligence packaging, limitations, demo script |
| License posture | Source-available proprietary; see `LICENSE`, `COMMERCIAL.md` |

---

## Capability snapshot (honest)

**Present (documented / tested in-repo to varying degrees):**

- Read-only project analysis and **prioritized port plans** (`analyze`, `port-plan`)
- HIP toolchain discovery on Windows and Linux ROCm layouts
- HIP-backed runtime subset (device/memory/stream/event) when built with HIP
- Bounded CUDA-syntax and PTX→AMD code-object paths
- Mixed-precision GEMM subset; optional FFT; small FP32 neural op set
- Explicit Python/C bindings; CPU-staged PyTorch module for four ops
- Wheel **admission/diagnostics** (not transparent CUDA wheel execution)

**Absent / not claimed:**

- Full CUDA API or Driver parity
- Automatic migration of arbitrary repositories
- Transparent CUDA binary or wheel execution
- `torch.cuda` / training / multi-GPU readiness
- Fabricated benchmarks or market metrics

Training profile text prints **NOT READY** for single- and multi-GPU training.
Keep that language in diligence materials.

---

## Suggested technical evaluation path (half-day)

1. Clone; create venv; `pip install -e .` and `pip install -e '.[inference]'` if demos needed.
2. `python -m compat doctor` and `python -m compat port-plan` on a sample CUDA tree.
3. On a HIP-capable machine, build with `COMPATCUDA_ENABLE_HIP=ON`, run CTest.
4. `compat init --self-test` / `compat demo --operation softmax` when library + HSACO exist.
5. Run `python -m unittest discover -s tests -v`.
6. Read `docs/acquisition/KNOWN_LIMITATIONS.md` and `COMPATIBILITY.md`.

Detailed command script: [`acquisition/BUYER_DEMO.md`](acquisition/BUYER_DEMO.md).

---

## Licensing and commercial contact

- Evaluation under root `LICENSE` (source-available proprietary).
- Production / redistribution / SaaS typically need a commercial license —
  [`COMMERCIAL.md`](../COMMERCIAL.md).
- Historical license transition notes: `LICENSE_TRANSITION_NOTICE.md`.
- Acquisition inquiries: **@theworker02** (same channel as commercial).

---

## Key documents map

| Document | Why read it |
| --- | --- |
| [ACQUISITION.md](../ACQUISITION.md) | Brief acquisition cover sheet |
| [acquisition/EXECUTIVE_SUMMARY.md](acquisition/EXECUTIVE_SUMMARY.md) | Diligence executive view |
| [acquisition/KNOWN_LIMITATIONS.md](acquisition/KNOWN_LIMITATIONS.md) | Defects and non-claims |
| [acquisition/BUYER_DEMO.md](acquisition/BUYER_DEMO.md) | Reproducible demo |
| [PORTING.md](../PORTING.md) | analyze → port-plan → HIPIFY loop |
| [drop-in-mode.md](drop-in-mode.md) | What DROP_IN means |
| [ai-compatibility.md](ai-compatibility.md) | AI profile honesty |
| [linux-developer-guide.md](linux-developer-guide.md) / [windows-deployment.md](windows-deployment.md) | Platform build notes |

---

## Risks diligence should weight highly

1. **Scope gap vs expectations** — buyers hearing “CUDA on AMD” may expect
   binary compatibility; this project refuses that framing.
2. **Platform verification skew** — Windows reference host vs Linux diversity.
3. **Third-party stack** — ROCm/HIP/driver versions are external dependencies.
4. **License transition** — understand historical grants vs current proprietary terms.
5. **Maintenance concentration** — single asserted maintainer contact `@theworker02`.

No valuation is stated in this document.

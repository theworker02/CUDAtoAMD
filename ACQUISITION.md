# Acquisition Brief — CUDAtoAMD

**Date:** 2026-09-22  
**Repository:** https://github.com/theworker02/CUDAtoAMD  
**Asserted copyright holder (from notices):** theworker02 (https://github.com/theworker02)  
**Package / release line (in-tree):** `cuda-amd-compat` **1.4.0**

**Status:** This document is a **buyer briefing and diligence index only**. **No acquisition, assignment, or exclusive license has occurred** by virtue of publishing or updating this file. Valuation, revenue, user counts, design wins, and CUDA parity percentages are **not stated** here.

---

## Executive summary

CUDAtoAMD is an **experimental, clean-room CUDA-oriented compatibility and native AMD runtime** for developers who need an **inspectable, fail-closed** path from CUDA-oriented source and tooling assumptions toward **HIP/ROCm execution on AMD hardware** — without impersonating NVIDIA GPUs, redistributing NVIDIA binaries, or claiming transparent CUDA binary or wheel drop-in.

The repository combines:

1. **Read-only diligence tooling** — CUDA source inventory, prioritized port plans, compatibility matrix rendering, toolchain discovery, wheel preflight, and release validation (Python `compat` CLI; pip package `cuda-amd-compat`).
2. **Bounded compilation paths** — narrow CUDA-syntax and validated PTX subsets lowered to AMD code objects (`.hsaco`) via explicit HIPIFY/hipcc invocation; not a general NVVM/PTX frontend or cubin translator.
3. **Native runtime subset** — CMake-built `compatcuda` library with clean-room CUDA Runtime/Driver-facing headers, explicit AMD C/C++ APIs, HIP adapter, and documented math/neural/graph subsets when ROCm dependencies are present.

The project is **source-available proprietary** (root `LICENSE`). Production, redistribution, SaaS, and typical commercial deployment require a **paid commercial license** (`COMMERCIAL.md`). Historical open-source grants for **prior public copies** are addressed in `LICENSE_TRANSITION_NOTICE.md` and are **not** automatically included in a proprietary transaction scope without legal review.

**Primary contact for acquisition and commercial inquiries:** GitHub [@theworker02](https://github.com/theworker02).

---

## Problem and market context (technical, not financial)

CUDA-oriented codebases routinely mix portable host logic with NVIDIA-specific APIs, build systems, prebuilt wheels, PTX/NVVM assumptions, and framework lock-in (`torch.cuda`, cuDNN-shaped calls, NCCL, etc.). Teams evaluating AMD HIP/ROCm often need more than a one-line “run HIPIFY” recommendation:

- **Visibility** into which symbols, headers, and constructs appear in a tree before committing port effort.
- **Prioritization** of NVIDIA-specific or unsupported patterns versus mappable surface.
- **Honest execution boundaries** — supported native AMD paths, explicit failures elsewhere — rather than silent emulation or spoofed device reports.

CUDAtoAMD targets that **developer boundary**: analyzer and database separate from execution so CI can run inventory without a GPU; runtime reports **actual AMD device properties**; unsupported calls return **deterministic errors** (including `cudaErrorNotSupported` when built without HIP).

This brief does **not** claim addressable market size, customer pipeline, or competitive win rate.

---

## Product definition (one paragraph)

CUDAtoAMD helps teams **assess** CUDA-oriented workloads, **compile supported** source/PTX paths to AMD code objects, and **execute native AMD code** through HIP/ROCm — **without pretending an AMD GPU is an NVIDIA GPU**. It ships a CUDA-facing host API **subset**, an explicit native AMD runtime, bounded source/PTX compilation routes, diagnostic tooling, and conservative wheel **admission** checks (not wheel **translation**).

---

## What is in the current release line (v1.4.0, per README)

Documented shipped surface includes:

| Area | Included (as documented) | Important boundary |
| --- | --- | --- |
| Device / runtime | HIP initialization, device properties, allocation, copies, streams, events, pools | Native HIP availability required |
| Modules | `.hsaco` file/memory load, symbol lookup, kernel launch | No cubin or arbitrary CUDA binary translation |
| CUDA-facing APIs | Documented Runtime and Driver subsets | Not complete CUDA ABI or replacement vendor DLL |
| Math | FP32/mixed-precision GEMM subset; optional C2C FFT | Not cuBLAS/cuFFT parity; hipFFT may be unavailable |
| Neural ops | RMSNorm, softmax, SwiGLU, interleaved RoPE | Correctness-first, not production-tuned kernels |
| Graphs | Single-stream native capture, instantiate, replay, cancellation | CUDA graph facade remains limited |
| Compiler | Bounded CUDA-syntax workflow and tested PTX subset | No general PTX/NVVM frontend or tensor-core translation |
| Python | CFFI lifecycle; CPU-staged NumPy/PyTorch inference paths | No autograd, `torch.cuda`, or transparent wheel support |
| Diagnostics | Source analysis, port-plan, matrix, toolchain, init, release checks | Describes **known scope**, not a compatibility percentage |

The compatibility database (`compatibility/cuda_api.json`, rendered as `COMPATIBILITY.md`) tracks **30** interface entries with statuses such as DIRECT, ADAPTED, PARTIAL, UNSUPPORTED, and NVIDIA_SPECIFIC. **`DIRECT` means a documented HIP analogue exists in inventory — not that ABI-level CUDA drop-in already works.**

---

## Architecture (transferable design)

High-level flow (from `ARCHITECTURE.md`):

```text
CUDA-oriented source → analyzer → compatibility database → report / HIPIFY handoff
                                             │
                                    future dispatcher
                                             │
                                HIP adapter → ROCm → AMD GPU
```

Design principles relevant to acquirers:

- **Analyzer isolation** — Python analyzer runs without ROCm; useful in CI for inventory-only gates.
- **Database as authority** — `compat matrix` and tests validate schema; runtime diagnostics are intended to align with the same IDs over time.
- **Fail-closed without HIP** — HIP-disabled builds exercise ABI/error paths; no fake successful GPU work.
- **Clean-room headers** — CUDA-shaped declarations are project-owned subsets for source experiments, not NVIDIA redistributables or DLL name substitution.
- **Truthful device reporting** — `compat init` reports real AMD name, architecture, memory, CU count, wave size; optional numerical self-test on supported hosts.

Repository layout (abbreviated):

```text
assets/          Project identity assets
bindings/        Explicit language bindings
compat/          Python CLI, analysis, compiler and wheel diagnostics
compatibility/   Capability inventory data
docs/            Developer, API, release, and acquisition documentation
docs/acquisition/  Diligence data room
docs/site/       GitHub Pages project site source
examples/        CUDA-oriented source and bounded PTX examples
include/         Public C, C++, and CUDA-facing compatibility headers
kernels/         Native HIP neural kernels
runtime/         Internal compatibility ABI contract
src/             Runtime, driver, graph, math, HIP adapter implementation
tests/           Python unittest suite and native CTests
```

---

## Deployment and distribution model

| Channel | Contents | Notes |
| --- | --- | --- |
| Git repository | Full source, docs, tests, examples | Primary distribution for native builds |
| PyPI name `cuda-amd-compat` | Analyzer and developer tools (wheel) | Does **not** bundle HIP/ROCm vendor DLLs or AMD GPU code objects |
| Native `compatcuda` | CMake output (`COMPATCUDA_ENABLE_HIP` on/off) | Requires local ROCm/HIP SDK when HIP enabled |
| GitHub Pages | `docs/site/` via Actions when enabled | Project site; URL typically `https://theworker02.github.io/CUDAtoAMD/` |

**Verified reference host (documented in README):** Windows with Visual Studio Build Tools, ROCm/HIP SDK 7.1, AMD RX 7800 XT (`gfx1101`). Linux ROCm paths are supported for **toolchain discovery**; not every test narrative is asserted on every Linux distribution.

**Explicit launch model:** `compat run`, `compat python` — **no DLL injection** or transparent interception of arbitrary CUDA vendor libraries.

---

## Technical differentiation (why this is not “another HIPIFY wrapper”)

1. **Inventory-first workflow** — `compat analyze` and `compat port-plan` produce read-only findings and prioritized actions; port-plan does **not** rewrite sources.
2. **Conservative wheel posture** — `wheel-doctor` and reviewed `compat python` admission block known CUDA-linked extension patterns; project does **not** claim CUDA wheel execution.
3. **Dual API surface** — CUDA-facing subset for bounded experiments plus explicit `amd_runtime.h` / C++ RAII for deterministic `.hsaco` execution.
4. **Bounded PTX** — documented subset with fail-closed rejection of untested operations (`docs/ptx-subset.md`).
5. **Acquisition-ready packaging** — data room, commercial/licensing docs, transition notice, and reproducible buyer demo script.

---

## Verification evidence (in-repo, narrow)

On the documented Windows HIP build, README states:

- **Six** native CTests pass with HIP enabled (including module correctness, CUDA contract, core HIP tests).
- **Four** deterministic ABI/error-path CTests pass with HIP disabled.
- Python unittest suite covers analysis, source/PTX boundaries, native lifecycle, module loading, mixed GEMM, FFT, graphs, neural operations, wheel admission, release metadata, and Pages site structure.
- Native vector-add fixture checked via file and memory module loading; `compat init --self-test` exercised on RX 7800 XT in current release narrative.

This is **narrow correctness evidence**, not production certification, not a benchmark suite, and not a claim about every AMD GPU, Linux variant, or CUDA application.

---

## Software stack and dependencies

| Layer | Technology | Buyer note |
| --- | --- | --- |
| Tooling | Python ≥ 3.10, setuptools; analyzer has no third-party deps | Editable install for diligence |
| Optional Python | `cffi`, `numpy`, user-provided PyTorch (`.[inference]`) | PyTorch not vendored as CUDA replacement |
| Native | C/C++, CMake, CTest | MSVC + NMake documented on Windows |
| GPU stack | AMD ROCm/HIP, hipBLAS; hipFFT optional | External, version-sensitive; not shipped in Python wheel |
| Migration tools | HIPIFY/hipcc invoked explicitly by `compat cc` | AMD toolchain components, not repackaged |

Third-party obligations: `THIRD_PARTY.md`, `LEGAL.md`. Security reporting: `SECURITY.md`.

---

## Roadmap posture (intent, not commitment)

From `ROADMAP.md`:

- **SHIPPED:** v1.4.x features listed above plus acquisition brief and data room.
- **NEXT:** Inventory sync (JSON ↔ exported symbols ↔ ABI docs); PARTIAL runtime gaps; Linux CI matrix; thin adapters only with conformance tests; diligence doc depth as deps change.
- **NON-GOAL (near term):** Transparent arbitrary CUDA binaries/wheels; NVIDIA spoofing; full NCCL/`torch.cuda` drop-in; claiming AMD hardware is CUDA hardware.

Acquirers should treat roadmap items as **maintainer intent**, not contractual deliverables, unless captured in a definitive agreement.

---

## What is typically included in a transaction (subject to definitive agreement)

Items commonly discussed for a **proprietary software asset sale or exclusive license** (wording for counsel):

| Asset class | Description |
| --- | --- |
| Git history and source | Full repository contents as of closing snapshot, including `src/`, `compat/`, `include/`, `kernels/`, `bindings/`, `tests/` |
| Documentation | Developer guides, compatibility matrix, architecture, acquisition data room, site source |
| Compatibility inventory | `compatibility/cuda_api.json` and generated `COMPATIBILITY.md` |
| Branding | Assets under `assets/` (e.g. `cudatoamd-logo.svg`); **trademark registration status UNKNOWN** — verify separately |
| Asserted copyright | Original works attributed to theworker02 in `NOTICE` / license files — **chain of title requires legal review** |
| Commercial/licensing framework | `COMMERCIAL.md`, `LICENSE`, transition notice as templates for successor enforcement |

**Not automatically included:** buyer infrastructure, ROCm/CUDA/PyTorch installations, cloud accounts, secrets, or fabricated operating metrics.

---

## What is explicitly NOT included or not claimed

- **Historical open-source grants** already received by third parties for **prior public releases** — see `LICENSE_TRANSITION_NOTICE.md`; editing the current `LICENSE` does not claw back those grants.
- **Third-party runtimes** — AMD ROCm/HIP, hipBLAS, hipFFT, drivers, NVIDIA CUDA toolkits/drivers/redistributables, PyTorch, NumPy (except as optional user installs).
- **Universal CUDA compatibility** — full Runtime/Driver/math/collective/DNN/RAND/SPARSE/SOLVER parity is **not** offered.
- **Transparent CUDA wheel or binary execution** — admission and blocking, not translation.
- **Production AI training** — AI profile documents training as **NOT READY**; framework integration is a bounded CPU float32 bridge for documented ops.
- **Customer lists, revenue, patents, exclusivity adjudication, or market share** — none asserted in this repository.

---

## Licensing and IP diligence checklist

1. Read root `LICENSE` (current proprietary evaluation terms).
2. Read `COMMERCIAL.md` (commercial inquiry process; no binding quote in-repo).
3. Read `LICENSE_TRANSITION_NOTICE.md` (historical vs post-transition copies; enforcement by holder or successor).
4. Read `LEGAL.md`, `THIRD_PARTY.md`, `NOTICE`.
5. Confirm **REQUIRES_LEGAL_REVIEW** for ownership, contributor agreements, and trademark.
6. No CLA/DCO program is described in-repo as of this brief’s date.
7. Asserted maintainer/contact concentration: single GitHub identity **theworker02**.

---

## Top technical diligence risks

1. **Expectation mismatch** — stakeholders equating “CUDA on AMD” with binary or wheel drop-in; project **refuses** that framing (`docs/drop-in-mode.md`, `KNOWN_LIMITATIONS.md`).
2. **External stack dependence** — behavior varies with ROCm/HIP/driver versions; hipFFT may be absent (deterministic backend-unavailable for FFT API).
3. **Platform evidence skew** — primary verified narratives emphasize Windows + specific GPU/SDK; Linux tooling exists but universal certification is **not** claimed.
4. **Inventory vs runtime gap** — `COMPATIBILITY.md` is an interface inventory; PARTIAL/ADAPTED entries need test validation before runtime claims.
5. **License transition** — mixing historical OSS trees with current proprietary code does not expand OSS rights to new material.
6. **Maintenance concentration** — bus factor and continuity planning should be explicit in any transaction.

---

## Suggested buyer evaluation path

**Half-day technical track** (detail in `docs/acquisition/BUYER_DEMO.md` and `docs/ACQUISITION_GUIDE.md`):

1. Clone; Python 3.10+ venv; `pip install -e .`
2. `python -m compat doctor`, `analyze`, `port-plan` on `examples/` or buyer-provided tree (read-only)
3. `python -m compat matrix`; review `COMPATIBILITY.md`
4. `python -m unittest discover -s tests -v` (CPU-side coverage)
5. Optional HIP machine: CMake HIP build, CTest, `compat init --self-test`, `compat demo --operation softmax`
6. Read `docs/acquisition/KNOWN_LIMITATIONS.md` and `PORTING.md`

**Data room index:** `docs/acquisition/README.md`

---

## Handoff and post-close integration notes (non-binding)

Acquirers often plan:

- Renaming or embedding `compatcuda` inside a larger GPU software stack while preserving clean-room boundaries.
- Extending `compatibility/cuda_api.json` and conformance tests before marketing runtime support.
- Commercial license migration for existing evaluators under successor terms.
- Separating **analyzer-only** CI products from **HIP runtime** SKUs.

No specific integration timeline or staffing is promised in this document.

---

## Related documents

| Document | Role |
| --- | --- |
| [docs/acquisition/EXECUTIVE_SUMMARY.md](docs/acquisition/EXECUTIVE_SUMMARY.md) | Short diligence overview |
| [docs/acquisition/KNOWN_LIMITATIONS.md](docs/acquisition/KNOWN_LIMITATIONS.md) | Limitations and non-claims |
| [docs/acquisition/BUYER_DEMO.md](docs/acquisition/BUYER_DEMO.md) | Reproducible commands |
| [docs/ACQUISITION_GUIDE.md](docs/ACQUISITION_GUIDE.md) | Extended technical buyer guide |
| [COMMERCIAL.md](COMMERCIAL.md) | Commercial license inquiries |
| [LICENSE_TRANSITION_NOTICE.md](LICENSE_TRANSITION_NOTICE.md) | Historical vs current license |
| [README.md](README.md) | Product and build documentation |
| [ROADMAP.md](ROADMAP.md) | Shipped / next / non-goals |

---

## Acquisition contact

**GitHub:** [@theworker02](https://github.com/theworker02)  
**Repository:** https://github.com/theworker02/CUDAtoAMD

Use the same channel for commercial licensing inquiries unless a separate process is agreed in writing.

**No valuation, revenue figure, or transaction timeline is stated in this document.**

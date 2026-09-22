# Acquisition Brief â€” CUDAtoAMD

**Date:** 2026-09-22  
**Repository:** https://github.com/theworker02/CUDAtoAMD  
**Default branch:** `main`  
**Primary language:** Python  
**Status:** Diligence briefing only. **No acquisition has occurred** by virtue of this file.  
**License:** Proprietary â€” sale, written commercial license, or completed asset transfer required (see root `LICENSE`).  
**Valuation:** Not stated.  
**Contact:** GitHub [@theworker02](https://github.com/theworker02) Â· [thanks.dev/u/gh/theworker02](https://thanks.dev/u/gh/theworker02)

> Cloning or forking this repository does **not** grant production, redistribution, SaaS, OEM, or commercial rights.

---

## 1. Executive thesis

<img src="assets/cudatoamd-logo.svg" width="720" alt="CUDAtoAMD Ã¢â‚¬â€ CUDA-oriented workflows, native AMD execution"> <a href="https://github.com/theworker02/CUDAtoAMD/releases"><img src="https://img.shields.io/github/v/release/theworker02/CUDAtoAMD?display_name=tag&sort=semver&label=release" alt="Latest release"></a> <a href="LICENSE"><img src="https://img.shields.io/github/license/theworker02/CUDAtoAMD" alt="source-available proprietary license"></a>

**Why a buyer cares:** CUDAtoAMD packages transferable product IP â€” source, docs, in-repo brand assets, and a diligence room under `docs/acquisition/` â€” under a clear proprietary posture so diligence can proceed without mistaking the repo for open source.

---

## 2. Product snapshot

| Item | Detail |
|------|--------|
| Product | CUDAtoAMD |
| Repo | `theworker02/CUDAtoAMD` |
| Language | Python |
| Open source? | **No** â€” proprietary |
| Rightsholder | theworker02 |
| Diligence pack | `docs/acquisition/` |

### Capability highlights (from current materials)

- Native HIP device initialization that reports the selected AMD GPU truthfully and can run a numerical GPU self-test.
- Clean-room CUDA Runtime and Driver API compatibility subsets, plus explicit AMD C/C++ APIs for streams, pools, events and HSACO modules.
- A bounded CUDA-syntax and PTX compilation route to AMD code objects through HIP tools.
- FP32/mixed-precision GEMM subsets, optional C2C FFT, selected neural operators, graph capture/replay, and explicit Python bindings.
- CUDA source inventory, toolchain discovery, wheel preflight, release validation and a GitHub Pages project site.
- The project is source-available proprietary licensed; see [LICENSE](LICENSE).
- It does not redistribute NVIDIA software, drivers or proprietary libraries; see [LEGAL.md](LEGAL.md) and [THIRD_PARTY.md](THIRD_PARTY.md).
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
- Validate untrusted source trees, PTX and wheel inputs before execution. The toolchain deliberately rejects unsupported constructs instead of attempting unsafe implicit conversions.
- Do not treat this runtime as a sandbox or a substitute for GPU-driver security updates.

---

## 3. Problem / opportunity

Teams evaluating CUDAtoAMD typically need either (a) a commercial right to run or embed it, or (b) outright ownership of the Product IP for strategic build-out. Public GitHub visibility without a proprietary license creates false assumptions about free production use. This brief and the linked data room make the commercial path explicit.

---

## 4. What ships today

Honest maturity: treat repository contents, README claims, tests, and release tags as the source of truth. Do not assume production customers, ARR, filed patents, or SLAs unless separately evidenced in diligence.

Typical transferable surfaces:

- Source tree and build/test scripts present in-repo
- Documentation and design notes
- Acquisition / diligence markdown under `docs/acquisition/`
- Branding assets committed to the repository (if any)

---

## 5. Demo / evaluation path (buyer)

Minimal path (no secrets required unless README says otherwise):

```
```text
CUDA-oriented source or explicit AMD code object
                    Ã¢â€â€š
                    Ã¢â€“Â¼
        CUDAtoAMD headers / developer tools
                    Ã¢â€â€š
                    Ã¢â€“Â¼
      compatibility ABI and semantic checks
                    Ã¢â€â€š
                    Ã¢â€“Â¼
             HIP / ROCm on AMD hardware
```
```powershell
python -m compat analyze path/to/cuda-project
python -m compat analyze path/to/cuda-project --format json
python -m compat port-plan path/to/cuda-project
python -m compat port-plan path/to/cuda-project --format json --output port-plan.json
python -m compat matrix
python -m compat doctor
```
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
```powershell
python -m compat toolchain --format json
python -m compat init --library build-hip/compatcuda.dll --self-test --format json
python -m compat doctor --native
```
```c
#include "amd_runtime.h"

```

Extended evaluation: `docs/acquisition/BUYER_EVALUATION.md`. Written NDA / evaluation grants may be required for private materials.

---

## 6. What a transaction typically includes

Subject to definitive schedules:

| Included (typical) | Excluded (typical) |
|--------------------|--------------------|
| Repo materials + asserted original IP | Seller personal accounts / unrelated repos |
| Docs + diligence room at closing | Third-party dependency source under separate licenses |
| In-repo brand marks as assigned | Secrets without rotation plan |
| Know-how captured in docs | Fabricated revenue, user, or adoption metrics |

---

## 7. Suggested deal structures

| Structure | When it fits |
|-----------|--------------|
| Non-exclusive commercial license | Deploy/run under seat or environment terms |
| Exclusive field-of-use license | Buyer wants exclusivity; seller may retain entity |
| Asset / IP assignment | Buyer wants ownership of Materials outright |
| OEM / redistribution | Separate agreement â€” not implied here |

Commercial terms (price, earnouts, escrow) are negotiated under NDA with counsel.

---

## 8. Buyer diligence checklist

- [ ] Confirm Rightsholder identity and authority to sell/license
- [ ] Inventory Materials (`docs/acquisition/ASSET_INVENTORY.md`)
- [ ] Review IP posture (`IP_PROVENANCE.md`) and dependencies (`DEPENDENCY_INVENTORY.md`)
- [ ] Run evaluation script (`BUYER_EVALUATION.md`)
- [ ] Review risks (`RISK_REGISTER.md`)
- [ ] Agree transfer scope (`TRANSFER_MANIFEST.md`) and handoff (`HANDOFF_CHECKLIST.md`)
- [ ] Supersede root `LICENSE` at closing via definitive agreement

---

## 9. Related documents

| Document | Purpose |
|----------|---------|
| `LICENSE` | Proprietary â€” no default grant |
| `docs/acquisition/README.md` | Data-room index |
| `docs/acquisition/EXECUTIVE_SUMMARY.md` | One-page thesis |
| `README.md` | Product overview |
| `SECURITY.md` | Vulnerability reporting |
| `COMMERCIAL.md` | Licensing contact path |
| `.github/FUNDING.yml` | Sponsors / thanks.dev |

---

## 10. Disclaimer

This package is informational and **does not** create a binding offer, grant of rights, or investment advice. Engage counsel for any transaction.

---

*Document version: 2.0.0 / 2026-09-22 Â· Classification: acquisition briefing*

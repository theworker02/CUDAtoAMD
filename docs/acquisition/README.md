# Acquisition Data Room — CUDAtoAMD

**Generated / updated:** 2026-09-22  
**Repository:** https://github.com/theworker02/CUDAtoAMD  
**Asserted copyright holder (from notices):** theworker02 (https://github.com/theworker02)  
**In-tree package version:** `cuda-amd-compat` 1.4.0

This directory is a **technical diligence data room**. It is factual, evidence-oriented, and intentionally discloses limitations. It is **not legal advice** and does **not** assert that an acquisition has occurred.

---

## Purpose

Buyers and internal diligence teams use this room to:

- Understand **what CUDAtoAMD is and is not** before technical deep dives.
- Run a **reproducible minimal demo** (CPU-only or with HIP hardware).
- Review **honest limitations** and licensing transition notes without fabricated metrics.

The root **[ACQUISITION.md](../../ACQUISITION.md)** is the primary **buyer brief** (cover sheet + index). This folder holds supporting diligence artifacts.

---

## Index

| Document | Purpose | Typical reader |
|----------|---------|----------------|
| [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) | Condensed buyer overview | Exec sponsor, corp dev |
| [KNOWN_LIMITATIONS.md](./KNOWN_LIMITATIONS.md) | Product, licensing, and third-party non-claims | Engineering + legal |
| [BUYER_DEMO.md](./BUYER_DEMO.md) | Copy-paste evaluation commands | Staff engineer |

**Broader technical guide (outside this folder):**

| Document | Purpose |
|----------|---------|
| [../ACQUISITION_GUIDE.md](../ACQUISITION_GUIDE.md) | Extended technical acquisition guide |
| [../../ACQUISITION.md](../../ACQUISITION.md) | Full acquisition brief |
| [../../COMMERCIAL.md](../../COMMERCIAL.md) | Commercial license inquiries |
| [../../LICENSE_TRANSITION_NOTICE.md](../../LICENSE_TRANSITION_NOTICE.md) | Historical OSS vs current proprietary |
| [../../COMPATIBILITY.md](../../COMPATIBILITY.md) | Interface inventory (not success rate) |
| [../../PORTING.md](../../PORTING.md) | analyze → port-plan → HIPIFY workflow |

---

## Recommended reading order

1. Root [ACQUISITION.md](../../ACQUISITION.md)  
2. [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)  
3. [KNOWN_LIMITATIONS.md](./KNOWN_LIMITATIONS.md)  
4. [BUYER_DEMO.md](./BUYER_DEMO.md) — run locally  
5. [../../COMPATIBILITY.md](../../COMPATIBILITY.md) and [../../PORTING.md](../../PORTING.md)  
6. Platform guides as needed: [../local-developer-guide.md](../local-developer-guide.md), [../linux-developer-guide.md](../linux-developer-guide.md), [../windows-deployment.md](../windows-deployment.md)  
7. AI / drop-in honesty: [../ai-compatibility.md](../ai-compatibility.md), [../drop-in-mode.md](../drop-in-mode.md)

---

## Diligence checklist (technical)

Use this as a starting agenda; adapt to your environment.

- [ ] Confirm Python 3.10+ and clone current `main`
- [ ] `pip install -e .` and `python -m compat doctor`
- [ ] Run analyze + port-plan on `examples/` or a **non-production** sample tree
- [ ] Review compatibility inventory statistics in `COMPATIBILITY.md` (30 tracked interfaces in current matrix)
- [ ] Run `python -m unittest discover -s tests -v`
- [ ] If HIP available: CMake HIP build, CTest, `compat init --self-test`
- [ ] Read license transition notice and commercial terms
- [ ] Document gaps vs your product expectations (especially wheels, training, full CUDA)

---

## Explicit non-claims for this data room

The following are **not** asserted anywhere in this data room:

- GitHub stars, download counts, active users, or revenue
- Patent portfolio or registered trademark status (verify independently)
- CUDA or framework parity percentages
- Manufacturer or cloud-provider endorsement
- That an acquisition or LOI exists

---

## Contact

Acquisition and commercial inquiries: GitHub [@theworker02](https://github.com/theworker02).

No valuation is stated in this document.

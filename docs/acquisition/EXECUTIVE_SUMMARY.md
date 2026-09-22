# Executive Summary — CUDAtoAMD

**Date:** 2026-09-21  
**Current license:** source-available proprietary (see root `LICENSE`)  
**Commercial inquiries:** [`COMMERCIAL.md`](../../COMMERCIAL.md)  
**Acquisition brief:** [`ACQUISITION.md`](../../ACQUISITION.md)

## What this is

An experimental clean-room **CUDA-oriented compatibility and native AMD
runtime** project: inventory CUDA usage, prioritize porting work, compile
bounded source/PTX subsets to AMD code objects, and run explicit HIP/ROCm
workloads — without spoofing NVIDIA hardware or claiming universal CUDA drop-in.

## Problem addressed

CUDA codebases mix portable logic with NVIDIA-specific APIs, binaries, wheels,
and framework assumptions. Teams need **diagnosable** AMD pathways and honest
failure modes more than silent emulation.

## Maturity

Package/version metadata **1.4.0** in-tree; positioned as an **experimental**
subset. Verified narratives emphasize a Windows + ROCm/HIP + RX 7800 XT
(`gfx1101`) reference host. Linux ROCm discovery is implemented for tooling;
do not assume every test has been reproduced on every distro.

## Deployment model

- `pip install` / editable install of the `compat` CLI and Python packages
- Optional CMake build of `compatcuda` with or without HIP
- Explicit environment launch (`compat run`, `compat python`) — no DLL injection

## Language / stack

- Python >= 3.10 (setuptools); optional `cffi`, `numpy`, user-provided PyTorch
- C/C++ runtime and headers; HIP/ROCm as external dependency
- CMake + CTest for native validation

## Licensing posture (factual)

- Current tree: proprietary / source-available terms in root `LICENSE`
- Commercial / production use: see `COMMERCIAL.md`
- Historical transition: see `LICENSE_TRANSITION_NOTICE.md`
- **REQUIRES_LEGAL_REVIEW** before treating ownership or exclusivity as adjudicated

## Ownership (asserted, not adjudicated)

Asserted holder/contact: **theworker02** (https://github.com/theworker02).  
No CLA/DCO program is described in-repo as of this date.

## What a buyer can expect

- Ability to evaluate analyzer, port-plan, docs, and (on suitable hardware) native demos
- Clear documentation of what is **not** implemented (wheels, training, full CUDA)
- Diligence materials that refuse fabricated metrics

## Top diligence risks

- Expectation mismatch vs “CUDA binary compatibility”
- Dependence on external ROCm/HIP/driver stacks
- License transition / third-party dependency obligations
- Single-maintainer concentration

No valuation is stated in this document.

# Acquisition Brief — CUDAtoAMD

**Date:** 2026-09-21  
**Status:** Briefing document only. **No acquisition has occurred** by virtue of this file.

## What the project does

Experimental clean-room CUDA→AMD compatibility tooling and native HIP/ROCm
runtime subset: source inventory, prioritized port plans, bounded compilers,
explicit AMD execution — without NVIDIA hardware impersonation.

## Problem

Teams need an inspectable, fail-closed path from CUDA-oriented code toward AMD
HIP/ROCm; full binary/wheel drop-in is widely assumed but rarely honest.

## What is included in a transaction (typical)

- Git repository and original CUDAtoAMD source/docs (subject to agreement)
- Asserted copyright in original works (subject to counsel / chain of title)
- Branding assets under `assets/` (registration status UNKNOWN)
- Acquisition data room under `docs/acquisition/`

## What is NOT included

- Historical open-source grants already received by third parties (see license transition notice)
- AMD ROCm/HIP, PyTorch, or other third-party runtimes
- NVIDIA CUDA toolkits, drivers, or redistributable binaries
- Buyer cloud accounts or secrets
- Fabricated user/revenue/CUDA-parity metrics (none claimed)

## Maturity

Experimental native AMD compatibility subset (package metadata 1.4.0). Not full
CUDA or transparent CUDA-wheel parity. Single asserted maintainer contact.

## Deployment model

pip-installable `compat` CLI + optional native HIP library build; Windows
reference host documented; Linux ROCm paths supported for toolchain discovery.

## Technical differentiation

Clean-room headers/runtime; truthful AMD device reporting; inventory + port-plan
prioritization; fail-closed unsupported paths; wheel admission without spoofing.

## Transferable IP / third-party / limitations

See `docs/acquisition/KNOWN_LIMITATIONS.md`, `docs/ACQUISITION_GUIDE.md`,
`COMMERCIAL.md`, `LICENSE_TRANSITION_NOTICE.md`.

## Handoff / evaluation

See `docs/acquisition/BUYER_DEMO.md` and `docs/acquisition/EXECUTIVE_SUMMARY.md`.

## Acquisition contact

GitHub [@theworker02](https://github.com/theworker02) · https://github.com/theworker02/CUDAtoAMD

No valuation is stated in this document.

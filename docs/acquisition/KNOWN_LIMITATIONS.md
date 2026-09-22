# Known Limitations — CUDAtoAMD

**Date:** 2026-09-21

## Product / technical

- **Not full CUDA:** Runtime/Driver/math coverage is a documented subset.
  `COMPATIBILITY.md` is an interface inventory, not an application-success rate.
- **Not automatic migration:** `compat port-plan` prioritizes findings; it does
  not rewrite trees or guarantee HIPIFY success.
- **Not transparent wheels:** `wheel-doctor` / `compat python` are admission and
  environment helpers. CUDA-linked extensions are blocked, not translated.
- **Not `torch.cuda`:** Framework integration is an explicit CPU float32 bridge
  for four neural ops; training is **NOT READY** (see `compat profile ai`).
- **PTX/NVVM:** Only a validated PTX subset is lowered; general PTX, tensor ops,
  and cubin/fatbin translation are out of scope.
- **Collectives / DNN / RAND / SPARSE / SOLVER:** Absent or partial inventory
  only — not productized parity.
- **Platform evidence skew:** Primary verified host narratives are Windows +
  specific GPU/SDK; Linux is supported for discovery/tooling but not asserted as
  universally certified.
- **DROP_IN:** Fail-closed ABI posture / planned profile name — not arbitrary
  app execution (see `docs/drop-in-mode.md`).

## Licensing / IP

- Source-available proprietary LICENSE; commercial terms via `COMMERCIAL.md`.
- Historical open-source copies (if any were distributed) are not clawed back by
  LICENSE edit alone — see `LICENSE_TRANSITION_NOTICE.md`.
- Asserted copyright contact `@theworker02`; chain of title requires legal review.
- No fabricated exclusivity or patent portfolio claims in this data room.

## Third-party

- AMD ROCm/HIP, hipBLAS, hipFFT, drivers — external, version-sensitive.
- Optional PyTorch/NumPy — user-installed; not vendored as a CUDA replacement.
- HIPIFY/hipcc are AMD toolchain components invoked explicitly.

## Do not assume

- GitHub stars, users, revenue, or design-win counts (not claimed here)
- Manufacturer or cloud-provider affiliation
- That `DIRECT` database rows mean ABI drop-in success
- That optional extras are production-hardened
- That this runtime is a security sandbox

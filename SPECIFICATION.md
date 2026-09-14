# Compatibility Specification 0.1

## Scope

The project supplies analysis, translation metadata, and eventually adapters to HIP/ROCm. It does not provide a CUDA ABI, NVIDIA device identity, PTX execution, or a claim of universal compatibility.

## Status values

`DIRECT` means a known HIP/ROCm operation is a candidate for direct mapping. `ADAPTED` requires argument, error, or capability adaptation. `TRANSLATED` requires source or IR translation. `EMULATED` requires a documented software implementation. `PARTIAL` supports only documented semantics. `REMOTE_CAPABLE` requires opt-in remote execution. `UNSUPPORTED` has no safe implementation. `NVIDIA_SPECIFIC` depends on NVIDIA-only behavior. `UNKNOWN` has not been classified.

All runtime claims require a conformance test. Database entries express a design classification, not proof that a drop-in ABI has shipped.

## Semantic and capability requirements

Providers must report each behavior as `native`, `translated`, `emulated`, or `unavailable`; physical AMD capability and compatibility-layer capability must remain distinct. A provider must fail with a structured compatibility error if it cannot satisfy a requested semantic requirement. It must never return success with reduced semantics.

## Provider contract (planned)

Each backend adapter will accept a normalized operation, validated arguments, a logical stream, and allocation ownership metadata. It returns either a result with execution strategy and semantic coverage or a stable error containing compatibility-entry ID, status, reason, and remediation. The first provider target is HIP; CPU and remote providers are optional future backends.

## Versioning

The database and analyzer output declare independent schema versions. Consumers must reject unsupported major schema versions. Compatibility data is additive within a minor release; status changes are documented in the changelog.

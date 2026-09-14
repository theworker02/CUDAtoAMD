# Compatibility Landscape

## Existing building blocks

AMD HIP is the primary portability layer: its API is CUDA-like and targets AMD GPUs through ROCm. HIPIFY already provides both a Clang-based source translator and a simpler pattern-based translator. The Clang path handles syntax, API calls, and launch differences but requires a compilable CUDA project and CUDA headers; the pattern-based path is less exact. HIPIFY itself cautions that it cannot seamlessly translate all CUDA code, libraries without an equivalent, or NVIDIA-specific performance work. [HIPIFY documentation](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html) [HIP porting guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)

ROCm exposes HIP-facing and ROCm-native math libraries. AMD's published HIPIFY surface includes Runtime, Driver, RTC, BLAS, Sparse, Solver, Random, FFT, DNN/MIOpen, Tensor, and CUB-family mappings. The published table is a valuable input, not proof of identical semantics for every API. [HIPIFY supported APIs](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/supported_apis.html)

## Where this project contributes

This project does not replace HIP, HIPIFY, Clang, or ROCm libraries. Its intended role is to make the compatibility decision explicit and testable:

- Maintain a small, versioned compatibility inventory with status, mapping, semantics, and strategy.
- Analyze a project before mutation, including CUDA headers, API symbols, launch syntax, inline assembly, NVIDIA compiler assumptions, and binary artifacts.
- Generate a transparent matrix rather than a misleading single compatibility score.
- Provide stable entry IDs that future runtime diagnostics and conformance results can reference.

## Known hard boundaries

PTX/cubin/fatbin artifacts are NVIDIA-targeted inputs and are not translated by this release. Inline PTX, exact device identity, CUDA IPC behavior, advanced memory behavior, and cooperative execution require feature-specific evaluation. A HIP analogue does not automatically preserve ABI, error, timing, memory, or numerical semantics. Remote CUDA execution, if ever added, must be explicit opt-in and cannot send allocations by default.

## Initial execution path

The proposed first tested backend path is source-level HIP migration for a small vector operation: allocation, host/device copies, kernel launch, synchronization, copy-back, and validation. Until a ROCm-capable machine runs this path with conformance tests, it remains planned rather than supported.

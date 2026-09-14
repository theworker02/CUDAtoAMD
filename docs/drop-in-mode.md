# Drop-In Mode: Current Contract

`DROP_IN` is a planned execution profile, not a claim that arbitrary CUDA applications can run today. This repository now ships an independently implemented **declaration subset** in `include/` and a `compatcuda` shared-library target with exported symbols.

Without a HIP backend, every state-changing operation fails explicitly with `cudaErrorNotSupported`; output pointers are nulled and device count is zero. This makes accidental deployment safe: applications cannot mistake an unavailable backend for a successful GPU operation.

When enabled and tested, the HIP backend must use a logical allocation registry, logical streams/events, normalized errors, and the project compatibility database. Direct HIP handles must not become part of the public CUDA-facing ABI.

CUDA launch syntax and device-language constructs require `hipcc`/Clang lowering. Headers alone cannot convert a conventional C++ compiler into a CUDA-language compiler.

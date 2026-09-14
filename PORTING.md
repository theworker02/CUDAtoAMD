# Porting Workflow

Run `python -m compat analyze .` before source changes. Treat `DIRECT` as a mapping candidate, not an automatic correctness guarantee. Resolve `NVIDIA_SPECIFIC`, `UNSUPPORTED`, and `UNKNOWN` findings before invoking HIPIFY. Preserve the original source and use HIPIFY in a separate working tree. Then build with `hipcc`, run functional tests, and record device, ROCm, compiler, and OS versions.

The current `compat migrate` and `compat run` commands are intentionally not exposed: no source-mutating or execution behavior has been implemented yet.

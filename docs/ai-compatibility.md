# AI Compatibility Profile

The machine-readable source is `spec/ai-compatibility.yaml`. This is a roadmap/status manifest, not a training-readiness certification.

The current core supports HIP-backed device, allocation, memory-copy, stream, and event dispatch when built with `COMPATCUDA_ENABLE_HIP=ON`. GEMM adapters, mixed precision, RNG, AI kernels, framework integration, and collectives remain unimplemented. Therefore neither `compat profile ai` nor any documentation may report single-GPU training as ready.

Windows has ROCm/HIP toolchain support in this environment, but the broader ROCm AI library and communication availability must be detected and separately validated per system.

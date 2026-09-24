# CUDAtoAMD — architecture

CUDAtoAMD is an experimental, clean-room runtime for **assessing** CUDA-oriented workloads and **executing** supported paths on AMD HIP/ROCm. It does not emulate NVIDIA hardware or run arbitrary closed CUDA binaries unchanged.

## Layered boundary

```text
CUDA-oriented source or explicit AMD code object
                    │
                    ▼
        CUDAtoAMD headers / developer tools
                    │
                    ▼
      compatibility ABI and semantic checks
                    │
                    ▼
             HIP / ROCm on AMD hardware
```

## Major subsystems

| Subsystem | Role |
| --- | --- |
| Host CUDA API subset | Bounded Runtime/Driver compatibility surface |
| Native AMD API | Streams, pools, events, HSACO modules |
| Compilation route | CUDA-syntax / PTX → AMD code objects via HIP tools |
| Operator kernels | GEMM subsets, FFT, selected neural ops, graph capture |
| Tooling | Source inventory, toolchain discovery, wheel preflight, release validation |
| Python bindings | Explicit, inspectable entry points (no silent emulation) |

## Safety posture

Unsupported behavior is reported explicitly. The project does not spoof `nvidia-smi`, redistribute NVIDIA binaries, or claim AMD GPUs are CUDA devices.

## Testing

Follow project `TESTING.md` and release validation scripts for your platform; Python and native tests vary by ROCm availability.

## Commercial

[../ACQUISITION.md](../ACQUISITION.md) · [acquisition/REPRODUCTION_COST.md](./acquisition/REPRODUCTION_COST.md)

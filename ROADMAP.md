# Roadmap — CUDAtoAMD

Statuses reflect intent on `main`. A feature is done when code, tests, and docs agree.

| Status | Meaning |
| --- | --- |
| **SHIPPED** | In the tree and usable |
| **NEXT** | Near-term, bounded |
| **LATER** | Valuable but not scheduled as committed work |
| **NON-GOAL** | Explicitly refused |

## SHIPPED (v1.4.x + acquisition-polish)

- Native HIP init + device truth reporting + optional numerical self-test
- CUDA Runtime/Driver compatibility subsets + explicit AMD C/C++ APIs
- Bounded CUDA-syntax / PTX → AMD code-object paths
- FP32 / mixed-precision GEMM, optional C2C FFT, neural ops, graph capture/replay
- Python bindings, wheel preflight, `compat analyze` / `doctor` / `port-plan`
- Acquisition brief + data room (`ACQUISITION.md`, `docs/acquisition/`)

## NEXT

- Authoritative inventory sync: `compatibility/cuda_api.json` ↔ exported symbols ↔ `docs/abi-symbols.md`
- Runtime gaps already classified PARTIAL: richer `cudaGetDeviceProperties`, managed memory, priority streams, `cudaGraphLaunch` facade over native graphs
- Linux CI verification matrix (ROCm host) alongside Windows-verified path
- Thin hipRAND or hipRTC adapter only when conformance tests exist
- Expand LEGAL / SECURITY / THIRD_PARTY diligence depth as deps change

## LATER

- Guided dry-run HIPIFY orchestration (`migrate` that never silently rewrites trees)
- Broader PTX subset (shared memory / barriers / signed types) for framework-generated PTX
- Stream-bound cuBLAS facade refinements

## NON-GOAL (near term)

- Transparent execution of arbitrary CUDA binaries or CUDA-locked wheels
- Spoofing `nvidia-smi`, inventing NVIDIA compute capability, or redistributing NVIDIA DLLs
- Claiming AMD hardware is CUDA hardware
- Full NCCL training parity or unmodified `torch.cuda` drop-in

See [`COMPATIBILITY.md`](COMPATIBILITY.md), [`docs/acquisition/KNOWN_LIMITATIONS.md`](docs/acquisition/KNOWN_LIMITATIONS.md), and [`CHANGELOG.md`](CHANGELOG.md).

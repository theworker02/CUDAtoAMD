# Asset inventory â€” CUDAtoAMD

## Repository surfaces

| Asset | Location / notes |
|-------|------------------|
| Source tree | Repository root / language packages |
| Tests | `test/`, `tests/`, CI workflows if present |
| Docs | `README.md`, `docs/` |
| Diligence room | `docs/acquisition/` |
| License / notices | `LICENSE`, transition notices if present |
| Funding | `.github/FUNDING.yml` |
| CI | `.github/workflows/` if present |
| Branding | logos/assets folders if present |

## Capability highlights

- Native HIP device initialization that reports the selected AMD GPU truthfully and can run a numerical GPU self-test.
- Clean-room CUDA Runtime and Driver API compatibility subsets, plus explicit AMD C/C++ APIs for streams, pools, events and HSACO modules.
- A bounded CUDA-syntax and PTX compilation route to AMD code objects through HIP tools.
- FP32/mixed-precision GEMM subsets, optional C2C FFT, selected neural operators, graph capture/replay, and explicit Python bindings.
- CUDA source inventory, toolchain discovery, wheel preflight, release validation and a GitHub Pages project site.
- The project is source-available proprietary licensed; see [LICENSE](LICENSE).
- It does not redistribute NVIDIA software, drivers or proprietary libraries; see [LEGAL.md](LEGAL.md) and [THIRD_PARTY.md](THIRD_PARTY.md).
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
- Validate untrusted source trees, PTX and wheel inputs before execution. The toolchain deliberately rejects unsupported constructs instead of attempting unsafe implicit conversions.
- Do not treat this runtime as a sandbox or a substitute for GPU-driver security updates.

## Usually excluded

Seller personal accounts, unrelated repos, and unreissued registry tokens â€” unless listed in the definitive agreement.

*Updated: 2026-09-22*

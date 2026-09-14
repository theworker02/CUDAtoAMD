# CUDA-wheel execution path

Transparent CUDA-wheel execution is not implemented. A normal CUDA Python wheel can contain compiled CPython extensions that import NVIDIA driver/runtime DLLs, CUDA device binaries, and framework-specific CUDA assumptions. Replacing those interfaces with false device reports or DLL names would be unreliable and conflicts with this project's clean-room, truthful-hardware contract.

Version 0.36.0 adds artifact identity to the release-safe boundary for this work: read-only wheel preflight and an explicit Python launcher can now require a reviewed SHA-256.

```powershell
python -m compat wheel-doctor path/to/package.whl --format json
python -m compat python --wheel path/to/package.whl --wheel-sha256 <64-lowercase-hex-digest> --library build-hip/compatcuda.dll --dry-run app.py
```

`wheel-doctor` never extracts or executes the archive. It applies bounded archive inspection, computes the full-file SHA-256, reads package metadata, identifies native extension files and CUDA device artifacts, and scans up to one MiB per native extension for common CUDA import names. Its result is a decision aid, not a security sandbox or a proof that a wheel will run.

`compat python` adds the discovered ROCm SDK `bin` directory to the child process path and can set `AMD_RUNTIME_LIB_PATH` for this project's explicit native bindings. With `--wheel`, it refuses a preflight-blocked/unverified native wheel before launching. With `--wheel-sha256`, it also requires the exact lowercase 64-digit SHA-256 produced by `wheel-doctor`; mismatches are blocked before launch. Put launcher options before the script path; everything after the script is passed to Python unchanged. It does **not** inject DLLs, rewrite imports, claim an AMD device is NVIDIA hardware, or make `torch.cuda` available.

## Current outcomes

| Preflight verdict | Meaning | Action |
| --- | --- | --- |
| `blocked` | Native extension imports CUDA libraries, CUDA device binaries, or NVIDIA runtime dependencies | Rebuild/port extension; do not launch as transparent CUDA |
| `unverified_native_extension` | Native code exists but no CUDA marker was found | Test explicitly; no automatic redirection exists |
| `ptx_review_required` | PTX is present without native CUDA binary blockers | Compile only the documented PTX subset to AMD code objects |
| `python_only_or_no_cuda_binary_detected` | No inspected CUDA-native component found | An explicit AMD Python launch may be appropriate; it does not add CUDA support |

The path toward broader framework execution is: PyTorch dispatcher/device and allocator integration, stream/event semantics, compiler and kernel coverage, ROCm-compatible packaging, and conformance tests. Each must be implemented and tested before a wheel can be admitted; none are substituted by the launcher.

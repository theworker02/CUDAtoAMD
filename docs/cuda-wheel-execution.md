# CUDA-wheel execution path

Transparent CUDA-wheel execution is **not implemented**. This document explains
why, what the admission tools actually do, and how to use them safely.

A normal CUDA Python wheel can contain:

- Compiled CPython extensions that import NVIDIA driver/runtime libraries
- CUDA device binaries (cubin/fatbin) or embedded PTX
- Framework assumptions that `torch.cuda` and NVIDIA packaging exist

Replacing those interfaces with false device reports or spoofed DLL names would
be unreliable and conflicts with this project’s clean-room, truthful-hardware
contract.

---

## What versioned tooling provides

Artifact identity and preflight (documented from the 0.33.0 / 0.36.0 line):

```powershell
python -m compat wheel-doctor path/to/package.whl --format json
python -m compat python --wheel path/to/package.whl --wheel-sha256 <64-lowercase-hex-digest> --library build-hip/compatcuda.dll --dry-run app.py
```

### `wheel-doctor` (read-only)

- Never extracts the archive to a durable workspace for execution
- Never runs extension code
- Applies bounded archive inspection
- Computes full-file SHA-256
- Reads package metadata
- Identifies native extension files and CUDA device artifacts
- Scans up to one MiB per native extension for common CUDA import names

**Result:** a decision aid — not a security sandbox, not a proof the wheel will
run on AMD, not an admission into a compatibility VM.

### `compat python` (explicit launcher)

- Adds the discovered ROCm SDK `bin` directory to the child `PATH`
- Can set `AMD_RUNTIME_LIB_PATH` for this project’s explicit native bindings
- With `--wheel`, refuses a preflight-blocked / unverified native wheel before
  launch
- With `--wheel-sha256`, requires the exact lowercase 64-digit digest from
  `wheel-doctor`
- Puts launcher options **before** the script path; remainder goes to Python

It does **not**:

- Inject DLLs
- Rewrite imports
- Claim an AMD device is NVIDIA hardware
- Make `torch.cuda` available
- Translate cubins inside the wheel

---

## Preflight verdicts

| Preflight verdict | Meaning | Action |
| --- | --- | --- |
| `blocked` | Native extension imports CUDA libraries, CUDA device binaries, or NVIDIA runtime dependencies | Rebuild/port extension; do not launch as transparent CUDA |
| `unverified_native_extension` | Native code exists but no CUDA marker was found | Test explicitly; no automatic redirection exists |
| `ptx_review_required` | PTX present without native CUDA binary blockers | Compile only the documented PTX subset to AMD code objects |
| `python_only_or_no_cuda_binary_detected` | No inspected CUDA-native component found | Explicit AMD Python launch may be appropriate; it does not add CUDA support |

Exit codes: `wheel-doctor` returns non-zero when blockers exist (CLI uses `2`
for blocked). Treat automation accordingly.

---

## Recommended workflow for third-party wheels

1. Compute and store the SHA-256 of the exact artifact under review.
2. Run `wheel-doctor --format json`; archive the JSON with the hash.
3. If `blocked`, stop — plan a ROCm/HIP rebuild of the extension or use a
   vendor ROCm wheel instead.
4. If pure-Python / no CUDA binary detected, you may use `compat python` as an
   environment helper only.
5. Never interpret a clean PATH launch as “CUDA compatibility achieved.”

Example dry-run gate:

```powershell
python -m compat python --wheel dist\pkg.whl --wheel-sha256 abc...def --dry-run smoke.py
```

---

## Path toward broader framework execution (not done)

Each of the following must be implemented and tested before a CUDA wheel could
be “admitted” in any strong sense — **none are substituted by the launcher**:

1. PyTorch (or other framework) dispatcher / device and allocator integration
2. Stream and event semantics visible to the framework
3. Compiler and kernel coverage for the ops the wheel needs
4. ROCm-compatible packaging of the extension itself
5. Conformance tests at framework scope

Until then, the honest product statement is: **admission control and
diagnostics**, not transparent CUDA-wheel execution.

---

## Relationship to source and PTX tools

| Tool | Wheel relevance |
| --- | --- |
| `compat analyze` / `port-plan` | Useful on **source** trees that build wheels |
| `compat ptx-object` | Only for documented PTX subset extracted/reviewed separately |
| `compat code-object` | Source kernels you control — not random wheel blobs |
| `compat run` | Explicit binary launch with ROCm PATH — still no CUDA spoofing |

---

## Security notes

- Untrusted wheels can still harm you if you install/import them outside these
  tools — preflight is not a sandbox.
- Hash pinning (`--wheel-sha256`) reduces accidental execution of a substituted
  artifact after review.
- See [SECURITY.md](../SECURITY.md) for vulnerability reporting.

---

## Related documentation

- [Framework integration](framework-integration.md)
- [Drop-in mode](drop-in-mode.md)
- [AI compatibility](ai-compatibility.md)
- [PORTING.md](../PORTING.md)
- [ACQUISITION_GUIDE.md](ACQUISITION_GUIDE.md)

# Porting Workflow

This guide describes a practical CUDA→AMD porting loop using CUDAtoAMD's
**read-only** inventory tools and external HIPIFY/hipcc. Nothing in this
document claims automatic migration or full CUDA parity.

## 1. Inventory the project (read-only)

```bash
python -m compat analyze path/to/cuda-project
python -m compat analyze path/to/cuda-project --format json
```

The analyzer scans sources and selected binary suffixes, classifies CUDA API
references against `compatibility/cuda_api.json`, and notes special constructs
(kernel launch syntax, inline PTX, NVIDIA compiler flags).

## 2. Produce a prioritized porting plan

```bash
python -m compat port-plan path/to/cuda-project
python -m compat port-plan path/to/cuda-project --format json
python -m compat port-plan path/to/cuda-project --output port-plan.txt
```

`compat port-plan` reuses `analyze()` and groups findings in this order:

1. **UNSUPPORTED** / **NVIDIA_SPECIFIC** (address or isolate first)
2. **UNKNOWN** (manual review — binaries, untracked symbols)
3. **PARTIAL**
4. **ADAPTED**
5. **DIRECT**

For each file it lists recommended actions and a coarse effort hint
(`low` / `medium` / `high`). Effort hints are status-based heuristics, not
measured engineering estimates.

**Caveats (enforced in the plan text):**

- This is advisory prioritization, **not** automatic migration.
- `DIRECT` is a mapping candidate, not an automatic correctness guarantee.
- Preserve original sources; run HIPIFY in a separate working tree.
- Precompiled CUDA device binaries are not translated by this project.
- Transparent CUDA-wheel / `torch.cuda` execution is not implemented.

## 3. Resolve blockers before HIPIFY

Treat `NVIDIA_SPECIFIC`, `UNSUPPORTED`, and unresolved `UNKNOWN` findings
before invoking HIPIFY. Inline PTX, NVVM atomics, IPC handles without a
mapping, and vendor device binaries typically need redesign or exclusion from
the AMD build.

## 4. Translate and build externally

1. Copy or branch sources into a separate tree.
2. Run AMD's `hipify-clang` (via `compat cc` when the ROCm SDK is discovered, or
   invoke HIPIFY/hipcc directly).
3. Build with `hipcc` / CMake against HIP/ROCm.
4. Run functional tests on the target GPU.

```bash
python -m compat doctor          # hipcc discovery (PATH, ROCM_PATH, /opt/rocm)
python -m compat toolchain --format json
python -m compat cc kernel.cu -o kernel.out   # explicit HIPIFY → hipcc pipeline
python -m compat run ./my_program --dry-run   # explicit ROCm PATH launch (no DLL injection)
```

## 5. Record evidence

After each successful port slice, record device name, gfx architecture, ROCm
version, compiler version, OS, and the tests that passed. Re-run
`compat analyze` / `compat port-plan` to track remaining inventory.

## Related commands

| Command | Role |
| --- | --- |
| `compat analyze` | Raw inventory report |
| `compat port-plan` | Prioritized actions + effort hints |
| `compat matrix` | Print/write `COMPATIBILITY.md` from the database |
| `compat run` | Explicit launch with ROCm `bin` prepended to `PATH` (no injection) |
| `compat cc` | Explicit HIPIFY → hipcc source pipeline |

See also [Linux developer guide](docs/linux-developer-guide.md),
[CUDA source compatibility](docs/cuda-source-compatibility.md), and
[ACQUISITION.md](ACQUISITION.md) for buyer-oriented packaging.

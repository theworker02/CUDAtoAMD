# Linux Developer Guide

**Audience:** developers building or evaluating CUDAtoAMD on Linux with AMD
ROCm/HIP.  
**Honesty note:** many project narratives and historical CTest evidence center
on a **Windows** reference host (RX 7800 XT / `gfx1101`). Linux is a first-class
target for the Python tooling and ROCm discovery paths described here, but you
must run builds and tests on your own distribution/GPU before claiming parity.

---

## What works without a GPU

The analyzer and most CLI diagnostics are pure Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m compat analyze path/to/cuda-project
python -m compat port-plan path/to/cuda-project
python -m compat matrix
python -m compat doctor
python -m unittest discover -s tests -v
```

No ROCm install is required for inventory, port-plan, matrix rendering, or the
majority of unit tests that do not load the native library.

---

## ROCm / hipcc discovery

`compat doctor` and `compat toolchain` resolve `hipcc` in this order:

1. `hipcc` on `PATH` (`shutil.which`)
2. `$ROCM_PATH/bin/hipcc` or `$HIP_PATH/bin/hipcc` when set
3. `/opt/rocm/bin/hipcc`
4. Versioned trees matching `/opt/rocm*/bin/hipcc` (newest sort first)

```bash
export ROCM_PATH=/opt/rocm
python -m compat doctor
python -m compat toolchain --format json
```

If doctor prints `hipcc: not found`, install your distro’s ROCm/HIP packages or
fix `PATH` / `ROCM_PATH` before attempting HIP builds or `compat cc`.

---

## Native library build (outline)

Exact package names vary by distribution and ROCm major version. Conceptually:

1. Install AMDGPU driver + ROCm user-space (HIP, hipcc, hipBLAS; hipFFT optional).
2. Install a C++ toolchain and CMake.
3. Configure with HIP enabled:

```bash
cmake -S . -B build-hip \
  -DCOMPATCUDA_ENABLE_HIP=ON \
  -DCOMPATCUDA_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}"
cmake --build build-hip -j"$(nproc)"
ctest --test-dir build-hip --output-on-failure
```

Disable FFT if hipFFT is unavailable:

```bash
cmake -S . -B build-hip -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_ENABLE_FFT=OFF ...
```

Set the Python loader path to your built shared object (name may be
`libcompatcuda.so` / `compatcuda.so` depending on CMake rules):

```bash
export AMD_RUNTIME_LIB_PATH="$PWD/build-hip/libcompatcuda.so"  # adjust to actual artifact
python -m compat init --library "$AMD_RUNTIME_LIB_PATH" --format json
python -m compat doctor --native
```

If `init` / native doctor fails, fix the library path and ROCm runtime linkage
before debugging higher-level demos.

---

## Neural demos and architecture

Neural kernels must be compiled for your GPU’s `gfx` target. Wrong architecture
→ missing kernel image errors. Pass the appropriate CMake arch flag used by this
repo (see `COMPATCUDA_TEST_ARCH` / neural target in `CMakeLists.txt`) and rebuild
`neural.hsaco` (or equivalent) for that GPU.

```bash
python -m pip install '.[inference]'
python -m compat demo --operation softmax --library "$AMD_RUNTIME_LIB_PATH"
```

Optional PyTorch CPU bridge: install torch separately, then use
`AmdInferenceModule` as documented in [framework-integration.md](framework-integration.md).

---

## Explicit launch helpers

```bash
python -m compat run --dry-run ./my_hip_binary
python -m compat python --dry-run ./script.py
python -m compat cc kernel.cu -o kernel.out -- -O2
```

- `run` / `python` prepend ROCm `bin` to `PATH` for the child process
- No DLL/`LD_PRELOAD` injection and no NVIDIA spoofing
- `cc` runs `hipify-clang` then `hipcc` from the discovered SDK root (Unix
  binaries without `.exe`)

---

## Porting workflow on Linux

```bash
python -m compat port-plan ~/src/my-cuda-app --output port-plan.txt
# address UNSUPPORTED / NVIDIA_SPECIFIC / UNKNOWN first
# HIPIFY in a separate tree, then hipcc
```

Details: [PORTING.md](../PORTING.md).

---

## Distro and multi-version tips

- Multiple ROCm versions under `/opt/rocm-X.Y` are common — set `ROCM_PATH`
  deliberately and record it in test logs.
- Mixing headers from one version with runtime from another is a frequent source
  of load failures; keep `CMAKE_PREFIX_PATH`, `ROCM_PATH`, and `LD_LIBRARY_PATH`
  aligned.
- Container images (where used) should pin ROCm and driver capabilities; this
  repo does not ship a canonical container.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `hipcc: not found` | SDK not installed or `PATH`/`ROCM_PATH` unset |
| Native library load error | Wrong `AMD_RUNTIME_LIB_PATH` or missing ROCm `.so` deps |
| No compatible kernel image | HSACO built for different `gfx` |
| FFT NOT_SUPPORTED | Built with FFT off or hipFFT missing |
| Wheel blocked | CUDA-linked extension — expected; see wheel docs |

---

## Related documentation

- [Windows deployment](windows-deployment.md) — verified Windows recipes
- [Local developer guide](local-developer-guide.md) — Windows-first day-one flow
- [Initialization](initialization.md)
- [Drop-in mode](drop-in-mode.md)
- [ACQUISITION_GUIDE.md](ACQUISITION_GUIDE.md)

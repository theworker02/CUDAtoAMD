# Buyer evaluation â€” CUDAtoAMD

## Goal

In 15â€“45 minutes, verify the Product builds or runs as documented and that proprietary notices are present.

## Steps

1. Confirm root `LICENSE` is proprietary and `ACQUISITION.md` exists.
2. Skim `README.md` install/run claims.
3. Execute:

```
```text
CUDA-oriented source or explicit AMD code object
                    Ã¢â€â€š
                    Ã¢â€“Â¼
        CUDAtoAMD headers / developer tools
                    Ã¢â€â€š
                    Ã¢â€“Â¼
      compatibility ABI and semantic checks
                    Ã¢â€â€š
                    Ã¢â€“Â¼
             HIP / ROCm on AMD hardware
```
```powershell
python -m compat analyze path/to/cuda-project
python -m compat analyze path/to/cuda-project --format json
python -m compat port-plan path/to/cuda-project
python -m compat port-plan path/to/cuda-project --format json --output port-plan.json
python -m compat matrix
python -m compat doctor
```
```bat
:: Backend-free ABI build: no GPU SDK required
cmake -S . -B build-msvc -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=OFF -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-msvc
ctest --test-dir build-msvc --output-on-failure

:: HIP build: HIP and hipBLAS required; hipFFT is optional
set CMAKE_PREFIX_PATH=C:\Program Files\AMD\ROCm\7.1
cmake -S . -B build-hip -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-hip
ctest --test-dir build-hip --output-on-failure
```
```powershell
python -m compat toolchain --format json
python -m compat init --library build-hip/compatcuda.dll --self-test --format json
python -m compat doctor --native
```
```c
#include "amd_runtime.h"

```

4. Run tests if present (`npm test`, `pytest`, `cargo test`, `go test ./...`, etc.).
5. Record README vs observed behavior gaps in workpapers.

## Pass criteria

- [ ] Clone succeeds
- [ ] Documented happy path works **or** failure is explained
- [ ] Minimal path needs no surprise secrets
- [ ] License notices intact

*Updated: 2026-09-22*

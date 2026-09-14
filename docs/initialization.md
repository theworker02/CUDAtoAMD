# Initialize and run on AMD hardware

Version 1.2.0 provides a direct initialization path for the implemented native runtime. It loads `compatcuda`, calls its HIP-backed `amdInit`, queries the real AMD device, and reports the selected architecture without presenting AMD hardware as NVIDIA hardware.

## One-time build

From an x64 Visual Studio developer prompt on Windows with the HIP SDK installed:

```powershell
cmake -S . -B build-hip -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\7.1"
cmake --build build-hip
```

## Initialize the runtime

```powershell
python -m compat toolchain --format json
python -m compat init --library build-hip/compatcuda.dll --device 0 --format json
```

On the verified development host, initialization reports the RX 7800 XT as `gfx1101` and selects HIP/ROCm as the backend. The command performs a real native library load and device query; it does not merely inspect the filesystem.

To run the built numerical kernel check after initialization:

```powershell
python -m compat init --library build-hip/compatcuda.dll --device 0 --self-test --format json
```

The self-test runs the FP32 softmax code object through AMD GPU memory and compares all output values with NumPy. It requires the `compatcuda_neural` build output, NumPy, and a GPU-compatible `neural.hsaco` in the native library directory. Failure is reported rather than treated as successful initialization.

## CUDA-oriented tooling that targets AMD

```powershell
python -m compat code-object examples/cuda_vector.cu -o vector.hsaco --arch gfx1101
python -m compat ptx-object examples/saxpy_bounded.ptx -o saxpy.hsaco --arch gfx1101
```

Both commands invoke HIP-Clang to produce native AMD code objects. `code-object` accepts the documented CUDA-syntax kernel subset; `ptx-object` accepts only the [validated PTX subset](ptx-subset.md). Use the [source workflow guide](cuda-source-compatibility.md) for Driver API module loading and launch, or the C/C++ headers in `include/` for host code.

## Boundaries

Initialization proves that this runtime can access the selected AMD HIP device. It does not make NVIDIA's CUDA driver available, convert arbitrary CUDA binaries, or transparently execute CUDA-only Python wheels. Use [wheel preflight](cuda-wheel-execution.md) before trying a Python package with native CUDA extensions.

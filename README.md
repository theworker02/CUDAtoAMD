<p align="center">
  <img src="assets/cudatoamd-logo.svg" width="720" alt="CUDAtoAMD — CUDA-oriented workflows, native AMD execution">
</p>

# CUDA-to-AMD Compatibility Runtime

An early, open interoperability project for assessing and incrementally adapting CUDA-oriented source workloads to AMD's HIP/ROCm ecosystem.

Version 1.3.0 adds an original project identity and a no-dependency [GitHub Pages site](docs/site/index.html), including a deploy workflow for GitHub Actions. It does not publish or enable Pages by itself; see [the Pages instructions](docs/site/README.md) after the repository has been pushed to GitHub.

Version 1.2.0 added a real [HIP/ROCm initialization workflow](docs/initialization.md): `compat init` loads the native runtime, initializes AMD hardware, reports the selected GPU truthfully, and can run a numerical GPU self-test. `compat toolchain` reports the HIP tools used to compile CUDA-oriented source for AMD.

The CUDA-facing host workflow also supports FP16/BF16/FP32 `cublasGemmEx`, strided-batched Ex GEMM and FP32 strided-batched GEMM (FP32 accumulation/output). Compile a CUDA-syntax kernel to an AMD code object, load and launch it through the Driver facade, and use the cuBLAS and cuFFT C2C subsets. See [the supported source workflow and its limits](docs/cuda-source-compatibility.md).

Local developers can also load native AMD kernels, run FP32 neural operators, replay captured GPU workflows, and call them explicitly from NumPy or CPU PyTorch inference code. The runtime provides C/C++ APIs, mixed-precision GEMM, optional FFT and a CUDA source inventory.

**[Start here: build, run a checked demo, and use the API](docs/local-developer-guide.md).**

This is an experimental native AMD runtime, **not** a universal CUDA replacement. It does not run arbitrary CUDA binaries or CUDA-locked frameworks unchanged.

After building and installing the inference extra:

```powershell
python -m compat doctor --native
python -m compat demo --operation softmax
```

## Quick start

The source analyzer requires Python 3.10+ and no third-party packages. GPU inference uses the optional inference extra and a separately built native library.

```powershell
python -m compat analyze path/to/cuda-project
python -m compat analyze path/to/cuda-project --format json
python -m compat matrix
python -m compat doctor
python -m compat toolchain --format json
python -m compat init --library build-hip/compatcuda.dll --self-test --format json
```

`analyze` is read-only. It reports detected CUDA constructs and their known compatibility classifications; it never modifies the project. See [COMPATIBILITY.md](COMPATIBILITY.md), [SPECIFICATION.md](SPECIFICATION.md), and [docs/compatibility-landscape.md](docs/compatibility-landscape.md).

## Native build environment

The ABI target was verified with Visual Studio Build Tools (C++ workload) and ROCm 7.1 on Windows. Start a 64-bit Visual Studio developer prompt, then:

```bat
cmake -S . -B build-msvc -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=OFF -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-msvc

set CMAKE_PREFIX_PATH=C:\Program Files\AMD\ROCm\7.1
cmake -S . -B build-hip -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_BUILD_TESTS=ON
cmake --build build-hip
```

The HIP-enabled build requires HIP and hipBLAS; hipFFT is optional (disable with COMPATCUDA_ENABLE_FFT=OFF). Native AMD graphs support single-stream capture, instantiation, replay and cancellation. The older CUDA graph facade still only creates/destroys handles. The HIP-disabled build needs no GPU SDK and returns deterministic not-initialized errors.

The experimental `compat cc` command invokes external HIPIFY/hipcc; its application-level workflow is not yet validated. `compat run --dry-run program.exe` shows the explicit ROCm-enabled launch environment; it does not inject or spoof DLLs. Runner options must precede the program path.

## Status

Verified locally on Windows with MSVC, HIP SDK 7.1 and gfx1101: a compiled native vector-add kernel produces 257 exact results through file and memory module loading; core GPU and small FP32 BLAS tests pass. This is narrow correctness evidence, not production readiness or a compatibility percentage.

Run `ctest --test-dir build-hip --output-on-failure` after building. The kernel fixture defaults to gfx1101; configure `-DCOMPATCUDA_TEST_ARCH=<your-gfx-target>` for another GPU. No matching device yields an explicit skipped kernel test. Other architectures and Linux have not been verified. For a staged local install use `cmake --install build-hip --prefix staging`; see [RELEASING.md](RELEASING.md) for the complete validation sequence.

See the [local developer guide](docs/local-developer-guide.md), [native runtime contract](docs/native-runtime.md), [Python bindings](docs/python-native.md) and [math adapters](docs/native-math.md). Native AMD handles now have type/lifetime validation. Neural support includes RMSNorm, softmax, SwiGLU and interleaved RoPE; framework support is an explicit CPU-staged inference bridge, without gradients. Full CUDA graph parity, transparent framework interception and validation of raw data pointers remain outside the implemented scope. Wheels do not bundle vendor DLLs or GPU code objects. Nothing has been published.

## License

Apache-2.0. This project does not redistribute NVIDIA software or libraries.

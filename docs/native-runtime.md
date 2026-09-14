# Native runtime contract (0.14.0)

The API accepts trusted, unbundled ELF64 AMDGPU code objects. It does not translate PTX, accept CUDA cubins, impersonate NVIDIA hardware or intercept binary imports. File loading uses the same bounded ELF validation as memory loading. Range checks catch malformed/truncated tables but do not make third-party GPU code safe.

## Build and verify

From an x64 Visual Studio developer prompt on the verified Windows host:

```bat
cmake -S . -B build-nohip-check -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=OFF
cmake --build build-nohip-check
ctest --test-dir build-nohip-check --output-on-failure
set CMAKE_PREFIX_PATH=C:\Program Files\AMD\ROCm\7.1
cmake -S . -B build-hip-check -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_TEST_ARCH=gfx1101
cmake --build build-hip-check
ctest --test-dir build-hip-check --output-on-failure
python -m unittest discover -s tests -v
```

The build compiles vector_add.hip with HIPCC using --genco and --no-gpu-bundle-output. The GPU test checks every output including a partial final block, both module loading paths, rejected malformed images, missing symbols, moved C++ owners, wrong-pool/double frees, live-pool destruction and explicit device selection while another device is current. A missing matching architecture is skipped, not counted as successful GPU execution.

## Ownership and synchronization

- Device handles are owned; stream/module/pool creation copies the device index. Their creation selects that device and restores the previous current device.
- Function handles are borrowed from a module. Module unload frees all function wrappers and invalidates them. Do not use or individually free them afterward.
- Native AMD opaque handles are checked by a type/lifetime registry before dereference. Host calls/destruction are serialized. Bounded tombstones prevent stale address reuse; this adds metadata and lock overhead. The implemented legacy CUDA stream/event/graph/context/module/function wrappers now also validate handle type/lifetime. Raw data pointers remain caller responsibilities; this is not complete CUDA validation.
- Pools serialize allocation/free and record ownership. Wrong-pool frees and repeated frees fail. Destruction returns AMD_ERROR_NOT_READY while live allocations remain.
- The historical initial_capacity parameter now means a HIP release-retention threshold. It does not reserve that many bytes up front.
- Pool destruction and module unload synchronize the device. These are conservative lifetime barriers, not zero-overhead operations.
- C++ DeviceBuffer retains native pool and stream owners across wrapper moves/destruction. Its destructor never throws and synchronizes after submitting a free. Use close() to observe cleanup failures. Destructors cannot report failures.
- Raw allocations still require explicit frees. Cross-stream consumers require explicit event ordering before free; host source/destination buffers must remain alive until transfers finish. There is no transparent host paging.

## Scope still incomplete

See the [local guide](local-developer-guide.md) for neural operators, HIP graph capture/replay, and explicit NumPy/PyTorch inference interop. Full CUDA graph parity, autograd, transparent CUDA-wheel execution and raw pointer validation are not implemented. The analyzer database describes possible mappings, not implemented ABI coverage. HIPBLAS remains required for the HIP-enabled build; hipFFT is optional. Linux and other GPU targets are unverified.

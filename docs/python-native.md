# Python native runtime (0.13.0)

Install from the repository with `python -m pip install ".[native]"`. The analyzer remains dependency-free; the native extra installs CFFI. Build the native library and kernel fixture as described in [native-runtime.md](native-runtime.md).

Set AMD_RUNTIME_LIB_PATH to the absolute compatcuda.dll (Linux: libcompatcuda.so) path. On Windows, AMD_RUNTIME_ROCM_ROOT may explicitly select a matching HIP SDK root. Discovery otherwise chooses one numerically latest installed SDK with a HIP DLL; it does not combine every SDK's binaries. Importing compat.native does not load HIP or initialize a GPU. Native libraries and AMD vendor dependencies are not distributed in the Python wheel.

Run the checked example:

```powershell
$env:AMD_RUNTIME_LIB_PATH = (Resolve-Path build-hip-check/compatcuda.dll).Path
python examples/python_vector_add.py build-hip-check/vector_add.hsaco
```

The example checks all results against a CPU reference. Its reported interval is an event measurement, not a performance benchmark or speedup claim.

## API and lifetime rules

- Runtime(path, rocm_root=...) selects a library; Device(index, runtime=...) uses it. Device.count(runtime=...) queries it.
- Device, Stream, MemoryPool, Event and Module support context managers and idempotent close(). Operations after close raise a structured RuntimeError with status and operation fields.
- MemoryPool.buffer(bytes, stream) returns an owned DeviceBuffer. Low-level allocate/free (also alloc_async/free_async) return raw addresses; raw allocations require explicit free.
- Buffers retain their pool/stream. Closing a pool or stream with live allocations fails. Close buffers first, pool next, stream last.
- copy_from_host_async accepts a contiguous buffer exporter. copy_to_host_async requires a writable one. Both retain the exporter until stream synchronization/query completion; callers must not mutate source bytes or read destination bytes before completion.
- copy_from_device_async supports same-stream buffers. Buffer-based kernel arguments also require their allocation stream. This conservative rule does not attempt to infer cross-stream dependencies.
- Module(device, path) and Module(device, image=bytes) accept trusted native ELF code objects. Functions are tied to their module and cannot be launched through a different or closed module.
- Kernel arguments must be DeviceBuffer or KernelArg with an explicit int32_t, uint32_t, int64_t, uint64_t, float or double type. This replaces unsafe guessing that every Python integer is a 64-bit device address.
- Event(device) creates on the requested device. Record, stream.wait_event and elapsed_ms are available; synchronize the relevant stream before reading elapsed time.
- Resources are not thread-safe at the Python layer: serialize access and destruction. Destructors attempt cleanup and warn on errors, but explicit context managers are required for predictable cleanup.

The old bindings/python/amd_runtime.py import path forwards to compat.native when the package is installed. Raw CFFI calls are an advanced escape hatch and are outside these lifetime guarantees.

## Verification

`python -m unittest discover -s tests -v` includes GPU tests if the built library, gfx1101 fixture and CFFI are present. Missing prerequisites are explicitly skipped; DLL load or initialization failures with present artifacts fail the tests.

Blas and FFTPlan provide the limited [native math adapters](native-math.md). The [local developer guide](local-developer-guide.md) covers Graph and the four neural operators, plus the explicit NumPy/PyTorch CPU-staged inference bridge. There is no transparent CUDA-wheel support or autograd.

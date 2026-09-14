# Start here: local AMD GPU development
This project is for developers who want to explicitly run their own native AMD
kernels, experiment with inference operators, or embed a small C/Python runtime.
You do not need an account or a cloud service. No drivers are replaced.

It is **not** a way to run an arbitrary CUDA wheel unchanged. PyTorch integration
here means a tested, explicit CPU-tensor inference bridge, not autograd or a new
PyTorch device backend.

## 1. Build once on Windows

Use an **x64 Native Tools Command Prompt for VS 2022**. Install a compatible AMD
driver and HIP SDK separately. The tested combination is HIP SDK 7.1 and RX 7800
XT (gfx1101). Substitute the actual gfx target for other GPUs; they are unverified.

```bat
set CMAKE_PREFIX_PATH=C:\Program Files\AMD\ROCm\7.1
cmake -S . -B build-hip-check -G "NMake Makefiles" -DCOMPATCUDA_ENABLE_HIP=ON -DCOMPATCUDA_TEST_ARCH=gfx1101
cmake --build build-hip-check
python -m pip install ".[inference]"
set AMD_RUNTIME_LIB_PATH=%CD%\build-hip-check\compatcuda.dll
python -m compat doctor --native
python -m compat demo --operation softmax
```

The build produces compatcuda.dll and neural.hsaco. The demo really executes
the GPU kernel and compares every output to a CPU reference. Success is not
printed unless the comparison passes. Other demos: rmsnorm, swiglu, rope.

In PowerShell, set the runtime path using:
```powershell
$env:AMD_RUNTIME_LIB_PATH = (Resolve-Path build-hip-check/compatcuda.dll).Path
```

## 2. Use NumPy or PyTorch

```python
import numpy as np
from compat.neural import Session

with Session() as amd:
    x = np.arange(32, dtype=np.float32).reshape(4, 8)
    probabilities = amd.numpy("softmax", x)
    normalized = amd.numpy("rmsnorm", x, np.ones(8, dtype=np.float32))
```

If PyTorch is already installed:
```python
import torch
from compat.neural import Session

with Session() as amd:
    x = torch.randn(4, 8, dtype=torch.float32)  # CPU, no gradients
    y = amd.torch("softmax", x)                 # AMD computation, CPU result
```

These convenience calls allocate/copy/synchronize. They are useful for embedding
and correctness experiments, not a promise to outperform framework-native
operators. Input must be finite contiguous FP32 data. Grad-enabled tensors are
rejected; there is no hidden detach or invented backward pass.

## 3. Keep data on the GPU and replay a graph

```python
from compat.native import Graph
from compat.neural import Session
import numpy as np

with Session() as amd:
    x = np.ones((4, 8), dtype=np.float32)
    with amd.pool.buffer(x.nbytes, amd.stream) as a, amd.pool.buffer(x.nbytes, amd.stream) as b:
        a.copy_from_host_async(x)
        amd.stream.synchronize()
        graph = Graph(amd.stream)
        try:
            with graph:  # capture exit instantiates; it does not destroy
                amd.ops.softmax(a, b, 4, 8)
            for _ in range(10):
                graph.launch()
            b.copy_to_host_async(x)
            amd.stream.synchronize()
        finally:
            graph.close()  # BEFORE buffers, modules, pools or streams close
```

Capture uses HIP graphs, not a Python list of callbacks. Allocate buffers and
resolve kernel symbols **before** capture. Replay uses the same addresses, so
upload new contents to existing buffers between launches. Kernel parameter
updates, multi-stream capture, explicit node editing and the complete CUDA
graph API are not implemented.

Graph lifetime conservatively pins all live native AMD resources at capture
start, including unrelated ones. Close all graphs before releasing resources.
This prevents module unload/free from making a later replay unsafe, but is more
restrictive than per-node dependency tracking.

## 4. Verification and troubleshooting

```bat
ctest --test-dir build-hip-check --output-on-failure
python -m unittest discover -s tests -v
```

- **DLL not found:** set AMD_RUNTIME_LIB_PATH. For mismatched vendor DLLs, set
  AMD_RUNTIME_ROCM_ROOT to the matching SDK directory. Do not copy random DLLs.
- **Neural code object missing:** build the compatcuda_neural target.
- **No compatible kernel image:** rebuild with your actual gfx architecture.
- **Resource closed / invalid handle:** never cast device addresses into handles;
  do not reuse handles after destroy.
- **Work pending / resource pinned:** close graphs before buffers and modules.
- **No HIP SDK:** the HIP-disabled build remains usable for ABI/error-path tests,
  but cannot execute GPU work.
- **Capability value -1:** unknown/not queried, not an assertion of support.
  The historical compute_units field forwards HIP's multiProcessorCount.

## Scope and safety

Native AMD device/stream/event/module/function/pool/BLAS/FFT/graph handles are
checked for type and lifetime. Calls and destruction are serialized by a host
lock. Retired wrapper records remain until process exit, capped at one million;
this prevents stale-address reuse but adds metadata and lock overhead.

Raw data pointers, arbitrary host pointers, array extents passed directly to C,
and malicious code objects are not made safe by handle validation. The implemented
CUDA-facing stream/event/graph/context/module/function wrappers also check handle
type/lifetime in 0.14.0. This is not a sandbox or a full CUDA implementation.
See the [CUDA source workflow](cuda-source-compatibility.md) for its tested subset.

The four neural kernels are portable correctness-first FP32 implementations:
RMSNorm and softmax currently use one thread per row. FlashAttention, convolution,
quantized neural kernels, autograd and transparent framework interception remain
outside this implemented scope. See [native math](native-math.md) for GEMM/FFT.

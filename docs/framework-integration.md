# Explicit PyTorch integration

**Status:** experimental, explicit, CPU-staged inference bridge  
**Introduced:** documented from 0.21.0 onward; `AmdInferenceModule` subclasses
`torch.nn.Module` when PyTorch is installed (0.24.0+)

This adapter exists so local experiments can call a small set of **native AMD**
operators from ordinary PyTorch CPU code without pretending to be `torch.cuda`.

---

## What you get

`compat.framework.AmdInferenceModule` is a closeable `torch.nn.Module` (when
`torch` is importable) around the tested native AMD operator session.

```python
import torch
from compat.framework import AmdInferenceModule

x = torch.randn(4, 128, dtype=torch.float32)  # CPU, finite, contiguous
with AmdInferenceModule("softmax", runtime=runtime) as operator:
    result = operator(x)  # CPU float32 result after AMD execution
```

Supported operation names (same contracts as `Session.torch`):

| Operation | Role |
| --- | --- |
| `softmax` | Row-wise softmax (correctness-first kernel) |
| `rmsnorm` | RMSNorm with weight tensor |
| `swiglu` | SwiGLU-style op per session contract |
| `rope` | Interleaved RoPE |

### Tensor contract

- Dtype: `torch.float32`
- Device: **CPU** inputs and outputs
- Layout: contiguous
- Values: finite
- Autograd: `requires_grad` must be false — grad-enabled tensors are **rejected**
- Lifecycle: module is closeable; calls after close fail

Staging is explicit: host → AMD GPU → host. Expect transfer overhead. This is
suitable for correctness experiments and embedding demos, not a claim of
framework-native performance.

---

## Environment and doctor

```powershell
python -m pip install ".[inference]"   # cffi + numpy; install torch yourself if needed
set AMD_RUNTIME_LIB_PATH=...\compatcuda.dll   # or .so on Linux
python -m compat framework-doctor --format json
python -m compat demo --operation softmax
```

`framework-doctor` reports factual capabilities (what is and is not supported).
Prefer its JSON over marketing language when writing integration notes.

Native library and matching `neural.hsaco` must exist for the GPU architecture.
See [local-developer-guide.md](local-developer-guide.md) and
[initialization.md](initialization.md).

---

## Composition patterns that are in scope

- Calling the module from ordinary `nn.Sequential`-style inference code **if**
  surrounding tensors stay on CPU float32 and no autograd is required
- Using `Session` NumPy APIs alongside the module for tests
- Capturing native graphs around lower-level session ops (see local guide) —
  graph APIs are native-AMD, not a CUDA graph facade completeness claim

## Patterns that are out of scope

| Pattern | Supported? |
| --- | --- |
| `tensor.cuda()` / `torch.device("cuda")` | No |
| Autograd / `loss.backward()` through the adapter | No |
| `torch.compile` / FX tracing assumptions | Not supported |
| Replacing PyTorch’s CUDA build with this DLL | No |
| Loading CUDA PyTorch wheels as if on NVIDIA | No — see wheel docs |
| Training loops, optimizers, GradScaler | No |
| Distributed DataParallel / NCCL | No |

A true transparent framework backend would need PyTorch dispatcher/device
integration, allocator and stream semantics, broad operator conformance,
packaging, and tests. **Those components are not present here.**

---

## Relationship to other entry points

| Entry point | Abstraction |
| --- | --- |
| `compat.neural.Session` | NumPy / optional torch helpers, explicit close |
| `AmdInferenceModule` | `nn.Module` ergonomics over the same native ops |
| C `amd_*` APIs | Lowest-level explicit control |
| CUDA-facing headers | Separate host subset; not a PyTorch backend |

---

## Failure modes you should plan for

- Missing `AMD_RUNTIME_LIB_PATH` or wrong DLL/so
- HSACO built for a different `gfx` architecture
- Non-contiguous or non-FP32 tensors
- Calling after `close()`
- Assuming silent CPU fallback — failures should surface as errors

---

## Acquisition / diligence note

When describing this feature to buyers or partners, use language like
**“explicit CPU-staged FP32 inference bridge for four operators”** — not
“PyTorch CUDA support” or “training on AMD via drop-in.”

See [ai-compatibility.md](ai-compatibility.md) and
[ACQUISITION_GUIDE.md](ACQUISITION_GUIDE.md).

# Explicit PyTorch integration (0.21.0)

The runtime now provides `compat.framework.AmdInferenceModule`, a real `torch.nn.Module` when PyTorch is installed, around the already-tested native AMD operator session. It is designed for local inference experiments where the caller wants a clear boundary between PyTorch tensors and native AMD execution.

```python
import torch
from compat.framework import AmdInferenceModule

x = torch.randn(4, 128, dtype=torch.float32)
with AmdInferenceModule("softmax", runtime=runtime) as operator:
    result = operator(x)
```

The selected native kernel runs through explicit CPU-to-AMD-GPU-to-CPU staging. Inputs and outputs are CPU, contiguous, finite `torch.float32` tensors with `requires_grad=False`. The module is closeable and rejects calls after it has closed. Supported operations are `softmax`, `rmsnorm`, `swiglu`, and interleaved `rope`; their shape contracts are the same as `Session.torch`. This supports composition in ordinary inference code, but not tracing, compilation or training.

Use the factual local report before integrating:

```powershell
python -m compat framework-doctor --format json
```

## Important boundary

This adapter does **not** replace `torch.cuda`, modify PyTorch installation files, load CUDA extensions, rewrite tensor device placement, support autograd, or run CUDA-only PyTorch wheels. A true transparent framework backend would need PyTorch dispatcher/device integration, allocator and stream semantics, compiler coverage, packaging and broad operator conformance. Those components are not present here.

The explicit adapter is useful when its transfer overhead and CPU tensor contract are acceptable. It should not be used to claim training support or performance parity with framework-native GPU execution.

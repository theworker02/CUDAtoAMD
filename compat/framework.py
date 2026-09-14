"""Explicit PyTorch inference integration for the native AMD operator subset.

This is intentionally not a replacement for torch.cuda or a CUDA-wheel loader.
It provides a small `torch.nn.Module` adapter whose supported operations run via
the checked local Session and whose unsupported training paths fail clearly.
"""
from .neural import Session

try:
    import torch
    _TorchModule = torch.nn.Module
except ImportError:  # Keep diagnostics/imports available without PyTorch.
    torch = None
    class _TorchModule:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


class AmdInferenceModule(_TorchModule):
    """A closeable, inference-only PyTorch module adapter.

    Construct after importing PyTorch. Inputs and outputs remain CPU float32
    tensors because transfers are explicit around the native AMD code object.
    """
    supported_operations = ("rmsnorm", "softmax", "swiglu", "rope")

    def __init__(self, operation="softmax", *, runtime=None, device=0, module_path=None, epsilon=1e-5):
        super().__init__()
        if operation not in self.supported_operations:
            raise ValueError(f"unsupported AMD inference operation: {operation}")
        if operation != "rmsnorm" and epsilon != 1e-5:
            raise ValueError("epsilon is only valid for rmsnorm")
        self.operation, self.epsilon = operation, epsilon
        self.session = Session(runtime=runtime, device=device, module_path=module_path)
        self.closed = False

    def forward(self, *inputs):
        if self.closed:
            raise RuntimeError("AMD inference module is closed")
        # Session.torch validates CPU, dtype, contiguous staging, and autograd.
        return self.session.torch(self.operation, *inputs, epsilon=self.epsilon)

    def close(self):
        if not self.closed:
            self.session.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def framework_report():
    """A factual, stable integration report suitable for scripts and CI."""
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {
        "pytorch_installed": torch_version is not None,
        "pytorch_version": torch_version,
        "integration": "explicit CPU-staged AMD inference module",
        "supported_operations": list(AmdInferenceModule.supported_operations),
        "supported_tensor_contract": "CPU, contiguous float32, requires_grad=False",
        "autograd": False,
        "torch_cuda_replacement": False,
        "cuda_wheel_interception": False,
    }

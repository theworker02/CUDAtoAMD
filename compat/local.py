"""Small local-developer diagnostics; no fabricated capability percentages."""
from pathlib import Path


def inspect_runtime(library=None):
    from .native import Runtime, Device
    runtime=Runtime(library)
    devices=[]
    for index in range(runtime.count()):
        with Device(index,runtime=runtime) as device:
            devices.append({"index":index,**device.properties})
    return {"library":str(runtime.path),"devices":devices,
            "neural_code_object":str(runtime.path.with_name("neural.hsaco")),
            "neural_file_present":runtime.path.with_name("neural.hsaco").is_file(),
            "execution_tested":False,
            "next_step":"python -m compat demo --operation softmax",
            "framework_mode":"explicit CPU-staged inference; not CUDA-wheel interception"}


def initialize(library=None, device=0, self_test=False, module=None):
    """Load the native ABI and initialize the real HIP/ROCm device backend."""
    from .native import Runtime, Device
    runtime = Runtime(library)  # amdInit executes while binding the ABI.
    total = runtime.count()
    if not total:
        raise RuntimeError("HIP initialized but no AMD GPU was reported by the runtime")
    with Device(device, runtime=runtime) as selected:
        properties = selected.properties
    result = {
        "initialized": True,
        "backend": "HIP/ROCm native AMD",
        "runtime_library": str(runtime.path),
        "devices_detected": total,
        "selected_device": device,
        "device": properties,
        "cuda_tooling_contract": "CUDA-facing source/header subset and explicit compiler; no NVIDIA driver emulation",
        "next_step": "python -m compat demo --operation softmax" if not self_test else None,
    }
    if self_test:
        result["self_test"] = demo("softmax", library, device, module)
    return result


def demo(operation="softmax", library=None, device=0, module=None):
    import numpy as np
    from .native import Runtime
    from .neural import Session
    runtime=Runtime(library)
    with Session(runtime=runtime,device=device,module_path=module) as session:
        x=np.linspace(-2,2,70,dtype=np.float32).reshape(7,10)
        if operation=="rmsnorm":
            weight=np.ones(10,dtype=np.float32)
            actual=session.numpy(operation,x,weight)
            expected=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)
        elif operation=="swiglu":
            actual=session.numpy(operation,x,x)
            expected=x/(1+np.exp(-x))*x
        elif operation=="rope":
            angles=np.full((7,5),.3,dtype=np.float32)
            cosine=np.cos(angles);sine=np.sin(angles)
            actual=session.numpy(operation,x,cosine,sine)
            expected=np.empty_like(x)
            expected[:,::2]=x[:,::2]*cosine-x[:,1::2]*sine
            expected[:,1::2]=x[:,::2]*sine+x[:,1::2]*cosine
        else:
            actual=session.numpy(operation,x)
            expected=np.exp(x-x.max(axis=-1,keepdims=True))
            expected/=expected.sum(axis=-1,keepdims=True)
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=2e-6)
        return {"operation":operation,"gpu":session.device.properties["name"],
                "elements_checked":int(actual.size),"max_absolute_error":float(np.max(np.abs(actual-expected))),
                "passed":True,"transfer_mode":"CPU -> AMD GPU -> CPU"}

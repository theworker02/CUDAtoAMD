"""Explicit, portable FP32 neural kernels and CPU-staged framework interop."""
from contextlib import ExitStack
import math
from pathlib import Path
from .native import Device, DeviceBuffer, KernelArg, MemoryPool, Module, Stream, _integer, _same_runtime


class NeuralOps:
    def __init__(self, device, module_path):
        self.runtime = device.runtime
        self.module = Module(device, module_path)
        try:
            self.functions = {name: self.module.get_function("amd_"+name)
                              for name in ("rmsnorm", "softmax", "swiglu", "rope")}
        except Exception:
            self.module.close()
            raise

    def _run(self, name, inputs, output, sizes, scalars, work):
        if not isinstance(output,DeviceBuffer):raise TypeError("output must be a DeviceBuffer")
        output._open()
        for buffer, size in zip((*inputs, output), sizes):
            if not isinstance(buffer, DeviceBuffer):
                raise TypeError("neural kernels require DeviceBuffer arguments")
            _same_runtime(self, buffer)
            buffer._open()
            if buffer.stream is not output.stream or buffer.size < size*4:
                raise ValueError("buffers must share a stream and fit the FP32 layout")
        if any(buffer.address == output.address for buffer in inputs):
            raise ValueError("neural output must not alias its inputs")
        self.module.launch(self.functions[name], ((work+127)//128,1,1), (128,1,1),
                           output.stream, [*inputs,output,*scalars])

    @staticmethod
    def _shape(rows, cols):
        rows=_integer(rows,"rows",(1<<31)-1,1)
        cols=_integer(cols,"cols",(1<<31)-1,1)
        _integer(rows*cols,"element count",(1<<31)-1,1)
        return rows,cols

    def rmsnorm(self, source, weight, output, rows, cols, epsilon=1e-5):
        rows,cols=self._shape(rows,cols)
        if not math.isfinite(epsilon) or epsilon<=0:
            raise ValueError("epsilon must be finite and positive")
        self._run("rmsnorm",(source,weight),output,(rows*cols,cols,rows*cols),
                  [KernelArg("int32_t",rows),KernelArg("int32_t",cols),KernelArg("float",epsilon)],rows)

    def softmax(self, source, output, rows, cols):
        rows,cols=self._shape(rows,cols)
        self._run("softmax",(source,),output,(rows*cols,rows*cols),
                  [KernelArg("int32_t",rows),KernelArg("int32_t",cols)],rows)

    def swiglu(self, gate, value, output, count):
        count=_integer(count,"count",(1<<31)-1,1)
        self._run("swiglu",(gate,value),output,(count,count,count),
                  [KernelArg("int32_t",count)],count)

    def rope(self, source, cosine, sine, output, rows, cols):
        rows,cols=self._shape(rows,cols)
        if cols%2:raise ValueError("interleaved RoPE requires an even last dimension")
        self._run("rope",(source,cosine,sine),output,(rows*cols,rows*cols//2,rows*cols//2,rows*cols),
                  [KernelArg("int32_t",rows),KernelArg("int32_t",cols)],rows*cols//2)

    def close(self):
        self.module.close()


class Session:
    """Local inference session. Framework transfers are explicit CPU staging.

    Default module location is next to the selected native library. Kernels must
    be built for the actual device architecture, not a guessed CUDA capability.
    """
    def __init__(self, *, runtime=None, device=0, module_path=None):
        self._resources=ExitStack()
        try:
            self.device=self._resources.enter_context(Device(device,runtime=runtime))
            self.stream=self._resources.enter_context(Stream(self.device))
            self.pool=self._resources.enter_context(MemoryPool(self.device))
            path=module_path or self.device.runtime.path.with_name("neural.hsaco")
            if not Path(path).is_file():
                raise FileNotFoundError(f"Neural code object missing: {path}. Build compatcuda_neural for {self.device.properties['gcn_arch']}.")
            self.ops=NeuralOps(self.device,path)
            self._resources.callback(self.ops.close)
        except Exception:
            self._resources.close()
            raise

    def numpy(self, operation, *inputs, epsilon=1e-5):
        import numpy as np
        if operation not in ("rmsnorm","softmax","swiglu","rope"):
            raise ValueError("operation must be rmsnorm, softmax, swiglu or rope")
        expected={"rmsnorm":2,"softmax":1,"swiglu":2,"rope":3}[operation]
        if len(inputs)!=expected:raise ValueError(f"{operation} requires {expected} inputs")
        for value in inputs:
            if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or not value.flags.c_contiguous:
                raise TypeError("inputs must be C-contiguous NumPy float32 arrays")
            if not value.size or not np.isfinite(value).all():
                raise ValueError("inputs must be nonempty and finite")
        source=inputs[0]
        if source.ndim<1:raise ValueError("source must have at least one dimension")
        cols=source.shape[-1];rows=source.size//cols
        if operation=="rmsnorm" and inputs[1].shape!=(cols,):
            raise ValueError("RMSNorm weight must match the last dimension")
        if operation=="swiglu" and inputs[1].shape!=source.shape:
            raise ValueError("SwiGLU inputs must have matching shapes")
        if operation=="rope" and (cols%2 or any(x.shape!=source.shape[:-1]+(cols//2,) for x in inputs[1:])):
            raise ValueError("RoPE cosine/sine must have half the source last dimension")
        output=np.empty_like(source)
        with ExitStack() as resources:
            buffers=[resources.enter_context(self.pool.buffer(x.nbytes,self.stream)) for x in inputs]
            result=resources.enter_context(self.pool.buffer(output.nbytes,self.stream))
            for buffer,value in zip(buffers,inputs):buffer.copy_from_host_async(value)
            if operation=="rmsnorm":self.ops.rmsnorm(*buffers,result,rows,cols,epsilon)
            elif operation=="softmax":self.ops.softmax(*buffers,result,rows,cols)
            elif operation=="swiglu":self.ops.swiglu(*buffers,result,source.size)
            else:self.ops.rope(*buffers,result,rows,cols)
            result.copy_to_host_async(output);self.stream.synchronize()
        return output

    def torch(self, operation, *inputs, **kwargs):
        import torch
        for tensor in inputs:
            if not isinstance(tensor,torch.Tensor) or tensor.device.type!="cpu" or tensor.dtype!=torch.float32 or tensor.requires_grad:
                raise TypeError("bridge requires CPU float32 inference tensors (requires_grad=False)")
        return torch.from_numpy(self.numpy(operation, *(t.contiguous().numpy() for t in inputs), **kwargs))

    def close(self):self._resources.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

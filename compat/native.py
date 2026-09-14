"""Explicit native AMD runtime bindings. Importing this module never loads HIP.

Install the 'native' extra for CFFI. Use context managers or close(); callers must
serialize operations on a Python resource and may load only trusted code objects.
"""
from __future__ import annotations

import os
import builtins
import re
import sys
import warnings
from pathlib import Path

import cffi

ffi = cffi.FFI()
ffi.cdef(Path(__file__).with_name("native.cdef").read_text(encoding="utf-8"))


class RuntimeError(builtins.RuntimeError):
    def __init__(self, status: int, operation: str, message: str):
        self.status, self.operation = status, operation
        super().__init__(f"{operation}: [{status}] {message}")


def _integer(value, name, maximum=(1 << 64) - 1, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _dims(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("dimensions must contain exactly three positive integers")
    return ffi.new("AmdDim3*", dict(zip(("x", "y", "z"),
        (_integer(v, "dimension", (1 << 32) - 1, 1) for v in value))))[0]


class Runtime:
    """One explicitly selected shared library; directory cookies live with it."""
    def __init__(self, library_path=None, *, rocm_root=None):
        suffix = "compatcuda.dll" if os.name == "nt" else "libcompatcuda.dylib" if sys.platform == "darwin" else "libcompatcuda.so"
        selected = library_path or os.environ.get("AMD_RUNTIME_LIB_PATH")
        if selected is None:
            candidate = Path(__file__).parents[1] / "build-hip-check" / suffix
            if not candidate.is_file():
                raise FileNotFoundError("Set AMD_RUNTIME_LIB_PATH to the built native runtime library")
            selected = candidate
        self.path = Path(selected).expanduser().resolve(strict=True)
        self._dll_directories = []
        if os.name == "nt":
            root = rocm_root or os.environ.get("AMD_RUNTIME_ROCM_ROOT")
            if root is None:
                base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm"
                candidates = [p for p in base.glob("*") if any((p / "bin").glob("amdhip64*.dll"))]
                root = max(candidates, key=lambda p: tuple(map(int, re.findall(r"\d+", p.name))), default=None)
            directories = [self.path.parent]
            if root:
                directories.append(Path(root).resolve(strict=True) / "bin")
            for directory in dict.fromkeys(directories):
                self._dll_directories.append(os.add_dll_directory(str(directory)))
        try:
            self.lib = ffi.dlopen(str(self.path))
        except OSError as exc:
            raise OSError(f"Cannot load {self.path}. Its HIP/hipBLAS dependencies must match; set AMD_RUNTIME_ROCM_ROOT to the SDK root. {exc}") from exc
        self.check(self.lib.amdInit(0), "amdInit")
        try:
            for symbol in ("amdEventCreateOnDevice", "amdBlasGemmStridedBatchedEx", "amdFftPlan1d", "amdGraphBegin"):
                getattr(self.lib, symbol)
        except AttributeError as exc:
            raise OSError("Native runtime is too old; rebuild version 0.13.0 or later") from exc

    def check(self, result, operation):
        if result:
            message = ffi.string(self.lib.amdGetErrorString(result)).decode("utf-8", "replace")
            raise RuntimeError(int(result), operation, message)

    def count(self):
        out = ffi.new("int*")
        self.check(self.lib.amdGetDeviceCount(out), "amdGetDeviceCount")
        return int(out[0])


_default = None


def default_runtime():
    global _default
    if _default is None:
        _default = Runtime()
    return _default


class _Resource:
    handle = ffi.NULL

    def _open(self):
        if self.handle == ffi.NULL:
            raise RuntimeError(5, type(self).__name__, "resource is closed")
        return self.handle

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if self.handle != ffi.NULL:
            try:
                self.close()
            except Exception as exc:
                warnings.warn(f"Resource cleanup failed: {exc}", ResourceWarning)

    def _call(self, name, *args):
        self.runtime.check(getattr(self.runtime.lib, name)(*args), name)


class Device(_Resource):
    def __init__(self, index=0, *, runtime=None):
        self.runtime = runtime or default_runtime()
        self.index = _integer(index, "device index", (1 << 31) - 1)
        out = ffi.new("AmdDevice*")
        self._call("amdGetDevice", out, self.index)
        self.handle = out[0]

    @classmethod
    def count(cls, runtime=None):
        return (runtime or default_runtime()).count()

    @property
    def properties(self):
        out = ffi.new("AmdDeviceProps*")
        self._call("amdGetDeviceProperties", self._open(), out)
        return {name: ffi.string(getattr(out, name)).decode("utf-8", "replace")
                for name in ("name", "gcn_arch")} | {
            name: int(getattr(out, name)) for name in (
                "total_vram_bytes", "compute_units", "max_threads_per_block",
                "native_wave_size", "supports_wmma", "is_cdna")}

    def close(self):
        if self.handle != ffi.NULL:
            self._call("amdDeviceDestroy", self.handle)
            self.handle = ffi.NULL


class Stream(_Resource):
    def __init__(self, device):
        self.runtime, self.device = device.runtime, device
        self._pending = []
        self._capturing = None
        self._allocations = set()
        out = ffi.new("AmdStream*")
        self._call("amdStreamCreate", device._open(), out)
        self.handle = out[0]

    def synchronize(self):
        self._call("amdStreamSynchronize", self._open())
        self._pending.clear()

    def ready(self):
        result = self.runtime.lib.amdStreamQuery(self._open())
        if result == 9:
            return False
        self.runtime.check(result, "amdStreamQuery")
        self._pending.clear()
        return True

    def wait_event(self, event):
        _same_runtime(self, event)
        self._call("amdStreamWaitEvent", self._open(), event._open())
        self._pending.append(event)
        if self._capturing is not None: self._capturing.append(event)

    def close(self):
        if self.handle != ffi.NULL:
            if self._allocations:
                raise RuntimeError(9, "Stream.close", "free all stream allocations first")
            self.synchronize()
            self._call("amdStreamDestroy", self.handle)
            self.handle = ffi.NULL


def _same_runtime(a, b):
    if a.runtime is not b.runtime:
        raise ValueError("resources belong to different Runtime instances")


class MemoryPool(_Resource):
    def __init__(self, device, capacity=0, *, initial_capacity_bytes=None):
        self.runtime, self.device = device.runtime, device
        self._allocations = {}
        if initial_capacity_bytes is not None:
            capacity = initial_capacity_bytes
        out = ffi.new("AmdMemPool*")
        self._call("amdMemPoolCreate", device._open(), _integer(capacity, "capacity"), out)
        self.handle = out[0]

    def allocate(self, bytes_count, stream):
        _same_runtime(self, stream)
        if self.device.index != stream.device.index:
            raise ValueError("pool and stream devices differ")
        out = ffi.new("AmdDeviceAddr*")
        self._call("amdMemAllocAsync", out, _integer(bytes_count, "bytes", minimum=1), self._open(), stream._open())
        address = int(out[0])
        self._allocations[address] = (bytes_count, stream)
        stream._allocations.add((id(self), address))
        return address

    alloc_async = allocate

    def free(self, address, stream):
        self._open()
        _same_runtime(self, stream)
        entry = self._allocations.get(address)
        if entry is None or entry[1] is not stream:
            raise ValueError("address must be freed by its owning pool and allocation stream")
        self._call("amdMemFreeAsync", address, self.handle, stream._open())
        del self._allocations[address]
        stream._allocations.discard((id(self), address))

    free_async = free

    def buffer(self, bytes_count, stream):
        return DeviceBuffer(self, bytes_count, stream)

    def close(self):
        if self.handle != ffi.NULL:
            if self._allocations:
                raise RuntimeError(9, "MemoryPool.close", "free all allocations first")
            self._call("amdMemPoolDestroy", self.handle)
            self.handle = ffi.NULL


class DeviceBuffer:
    address = 0

    def __init__(self, pool, bytes_count, stream):
        self.pool, self.stream, self.runtime = pool, stream, pool.runtime
        self.size = _integer(bytes_count, "bytes", minimum=1)
        self.address = pool.allocate(self.size, stream)

    def _open(self):
        if not self.address or self.address not in self.pool._allocations:
            raise RuntimeError(5, "DeviceBuffer", "buffer is closed")
        self.stream._open()
        return self.address

    def copy_from_host_async(self, source):
        view = ffi.from_buffer(source)
        size = ffi.sizeof(view)
        if size > self.size:
            raise ValueError("source exceeds device buffer")
        self.runtime.check(self.runtime.lib.amdMemcpyHtoDAsync(
            self._open(), view, size, self.stream._open()), "amdMemcpyHtoDAsync")
        self.stream._pending.append(view)
        if self.stream._capturing is not None: self.stream._capturing.extend((self, view))

    def copy_to_host_async(self, destination):
        view = ffi.from_buffer(destination, require_writable=True)
        size = ffi.sizeof(view)
        if size > self.size:
            raise ValueError("destination exceeds device buffer")
        self.runtime.check(self.runtime.lib.amdMemcpyDtoHAsync(
            view, self._open(), size, self.stream._open()), "amdMemcpyDtoHAsync")
        self.stream._pending.append(view)
        if self.stream._capturing is not None: self.stream._capturing.extend((self, view))

    def copy_from_device_async(self, source, bytes_count=None):
        _same_runtime(self, source)
        if source.stream is not self.stream:
            raise ValueError("device copies require the same stream")
        size = source.size if bytes_count is None else _integer(bytes_count, "bytes")
        if size > min(self.size, source.size):
            raise ValueError("copy exceeds device buffer")
        self.runtime.check(self.runtime.lib.amdMemcpyDtoDAsync(
            self._open(), source._open(), size, self.stream._open()), "amdMemcpyDtoDAsync")
        if self.stream._capturing is not None: self.stream._capturing.extend((self,source))

    def close(self):
        if self.address:
            self.pool.free(self.address, self.stream)
            self.address = 0
            self.stream.synchronize()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if self.address:
            try:
                self.close()
            except Exception as exc:
                warnings.warn(f"Buffer cleanup failed: {exc}", ResourceWarning)


class Event(_Resource):
    def __init__(self, device=None):
        device = device or Device()
        self.runtime, self.device = device.runtime, device
        out = ffi.new("AmdEvent*")
        self._call("amdEventCreateOnDevice", device._open(), out)
        self.handle = out[0]

    def record(self, stream):
        _same_runtime(self, stream)
        if self.device.index != stream.device.index:
            raise ValueError("event and stream devices differ")
        self._call("amdEventRecord", self._open(), stream._open())
        if stream._capturing is not None: stream._capturing.append(self)

    def elapsed_ms(self, stop):
        _same_runtime(self, stop)
        if self.device.index != stop.device.index:
            raise ValueError("event devices differ")
        out = ffi.new("float*")
        self._call("amdEventElapsedTime", out, self._open(), stop._open())
        return float(out[0])

    def close(self):
        if self.handle != ffi.NULL:
            self._call("amdEventDestroy", self.handle)
            self.handle = ffi.NULL


class KernelArg:
    """Exact scalar ABI type; Python integers are never guessed as kernel types."""
    TYPES = {"int32_t", "uint32_t", "int64_t", "uint64_t", "float", "double"}

    def __init__(self, ctype, value):
        if ctype not in self.TYPES:
            raise ValueError(f"unsupported kernel argument type: {ctype}")
        self.value = ffi.new(ctype + "*", value)


class Function:
    def __init__(self, module, handle):
        self.module, self.handle = module, handle


class Module(_Resource):
    def __init__(self, device, path=None, *, image=None):
        self.runtime, self.device = device.runtime, device
        if (path is None) == (image is None):
            raise ValueError("provide exactly one module path or image")
        out = ffi.new("AmdModule*")
        if image is not None:
            view = ffi.from_buffer(image)
            self._call("amdModuleLoadData", device._open(), view, ffi.sizeof(view), out)
        else:
            encoded = os.fsencode(path)
            if b"\0" in encoded:
                raise ValueError("module path contains NUL")
            self._call("amdModuleLoadFile", device._open(), encoded, out)
        self.handle = out[0]

    def get_function(self, name):
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("kernel name must be a nonempty string without NUL")
        out = ffi.new("AmdFunction*")
        self._call("amdModuleGetFunction", out, self._open(), name.encode("utf-8"))
        return Function(self, out[0])

    def launch(self, function, grid, block, stream, kernel_args, shared_mem_bytes=0):
        self._open()
        _same_runtime(self, stream)
        if not isinstance(function, Function) or function.module is not self:
            raise ValueError("function belongs to another module")
        if stream.device.index != self.device.index:
            raise ValueError("module and stream devices differ")
        values = []
        for arg in kernel_args:
            if isinstance(arg, DeviceBuffer):
                if arg.stream is not stream:
                    raise ValueError("kernel buffers must use their allocation stream")
                values.append(ffi.new("AmdDeviceAddr*", arg._open()))
            elif isinstance(arg, KernelArg):
                values.append(arg.value)
            else:
                raise TypeError("kernel arguments must be DeviceBuffer or KernelArg")
        pointers = ffi.new("void*[]", [ffi.cast("void*", value) for value in values]) if values else ffi.NULL
        self._call("amdLaunchKernel", function.handle, _dims(grid), _dims(block),
                   _integer(shared_mem_bytes, "shared memory", (1 << 32) - 1), stream._open(), pointers)
        stream._pending.append((self, values, pointers))
        if stream._capturing is not None: stream._capturing.extend((self, *kernel_args, values, pointers))

    def close(self):
        if self.handle != ffi.NULL:
            self._call("amdModuleUnload", self.handle)
            self.handle = ffi.NULL


class Blas(_Resource):
    """Column-major GEMM. F16/BF16 input storage; FP32 accumulation and output."""
    def __init__(self, device):
        self.runtime, self.device = device.runtime, device
        out = ffi.new("AmdBlasHandle*")
        self._call("amdBlasCreateOnDevice", device._open(), out)
        self.handle = out[0]

    def gemm(self, a, b, c, *, m, n, k, input_type="f32", trans_a=False,
             trans_b=False, alpha=1.0, beta=0.0, lda=None, ldb=None, ldc=None,
             batches=1, stride_a=None, stride_b=None, stride_c=None):
        types = {"f32": (0,4), "f16": (1,2), "bf16": (2,2)}
        if input_type not in types:
            raise ValueError("input_type must be f32, f16 or bf16")
        dtype, width = types[input_type]
        m,n,k = (_integer(v, "matrix dimension", (1<<31)-1, 1) for v in (m,n,k))
        batches = _integer(batches, "batches", (1<<31)-1, 1)
        if not isinstance(trans_a, bool) or not isinstance(trans_b, bool):
            raise ValueError("transpose flags must be booleans")
        lda = _integer(lda if lda is not None else k if trans_a else m, "lda", (1<<31)-1, 1)
        ldb = _integer(ldb if ldb is not None else n if trans_b else k, "ldb", (1<<31)-1, 1)
        ldc = _integer(ldc if ldc is not None else m, "ldc", (1<<31)-1, 1)
        spans = (lda*(m if trans_a else k), ldb*(k if trans_b else n), ldc*n)
        strides = tuple(_integer(v if v is not None else span, "stride", (1<<63)-1)
                        for v,span in zip((stride_a,stride_b,stride_c), spans))
        stream = c.stream
        for buffer,span,stride,bytes_per_value in zip((a,b,c),spans,strides,(width,width,4)):
            _same_runtime(self, buffer)
            buffer._open()
            if buffer.stream is not stream:
                raise ValueError("GEMM buffers must share a stream")
            if (span + (batches-1)*stride)*bytes_per_value > buffer.size:
                raise ValueError("GEMM layout exceeds buffer")
        if c.address in (a.address, b.address):
            raise ValueError("GEMM output must not alias its inputs")
        self._call("amdBlasGemmStridedBatchedEx", self._open(), stream._open(), dtype,
                   int(trans_a), int(trans_b), m,n,k, alpha, a.address,lda,strides[0],
                   b.address,ldb,strides[1], beta,c.address,ldc,strides[2],batches)
        if stream._capturing is not None: stream._capturing.extend((self,a,b,c))

    def close(self):
        if self.handle != ffi.NULL:
            self._call("amdBlasDestroy", self.handle)
            self.handle = ffi.NULL


class FFTPlan(_Resource):
    """Batched contiguous complex FP32 FFT; inverse output is not normalized."""
    def __init__(self, device, length, batches=1):
        self.runtime, self.device = device.runtime, device
        self.length = _integer(length, "FFT length", (1<<31)-1, 1)
        self.batches = _integer(batches, "FFT batches", (1<<31)-1, 1)
        out = ffi.new("AmdFftPlan*")
        self._call("amdFftPlan1d", device._open(), self.length, self.batches, out)
        self.handle = out[0]

    def execute(self, source, destination, *, inverse=False):
        for buffer in (source, destination):
            _same_runtime(self, buffer)
            buffer._open()
            if buffer.size < self.length*self.batches*8:
                raise ValueError("FFT buffer too small")
        if source.stream is not destination.stream:
            raise ValueError("FFT buffers must share a stream")
        if not isinstance(inverse, bool):
            raise ValueError("inverse must be boolean")
        self._call("amdFftExecC2C", self._open(), source.stream._open(),
                   source.address, destination.address, 1 if inverse else -1)
        if source.stream._capturing is not None: source.stream._capturing.extend((self,source,destination))

    def close(self):
        if self.handle != ffi.NULL:
            self._call("amdFftDestroy", self.handle)
            self.handle = ffi.NULL


class Graph(_Resource):
    """Capture a preallocated single-stream workflow and replay it on the GPU.

    Use 'with Graph(stream) as graph:' to capture, then graph.launch().
    Unlike ordinary resource context managers, capture exit instantiates but
    does not close the graph. Always close it before closing captured resources.
    """
    def __init__(self, stream):
        self.stream, self.runtime = stream, stream.runtime
        self._keepalive = []
        self._active = False

    def __enter__(self):
        if self.handle != ffi.NULL or self.stream._capturing is not None:
            raise ValueError("graph capture already started")
        self.stream.synchronize()
        out = ffi.new("AmdGraph*")
        self._call("amdGraphBegin", self.stream._open(), out)
        self.handle = out[0]
        self.stream._capturing = self._keepalive
        self._active = True
        return self

    def __exit__(self, kind, value, traceback):
        self.stream._capturing = None
        if kind is not None:
            self.close()
            return False
        try:
            self._call("amdGraphEnd", self._open())
        except Exception:
            self.close()
            raise
        self._active = False

    def launch(self):
        self._call("amdGraphLaunch", self._open(), self.stream._open())

    def close(self):
        if self.handle != ffi.NULL:
            self._call("amdGraphDestroy", self.handle)
            self.handle = ffi.NULL
            self._active = False
            self.stream._capturing = None
            self._keepalive.clear()

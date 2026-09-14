"""Real GPU binding integration tests; unavailable build artifacts are skipped."""
import array
import cmath
import math
import struct
import gc
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

if importlib.util.find_spec("cffi") is not None:
    from compat.native import Blas, FFTPlan, Device, Event, KernelArg, MemoryPool, Module, Runtime, RuntimeError, Stream
else:
    Runtime = None

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build-hip-check" / ("compatcuda.dll" if os.name == "nt" else "libcompatcuda.so")
IMAGE = ROOT / "build-hip-check" / "vector_add.hsaco"


@unittest.skipIf(Runtime is None, "install native extra for CFFI")
class LoaderTests(unittest.TestCase):
    def test_import_does_not_load_runtime(self):
        env = dict(os.environ, AMD_RUNTIME_LIB_PATH="certainly-missing-library")
        subprocess.run([sys.executable, "-c", "import compat.native"], env=env, check=True)

    def test_missing_library_is_actionable(self):
        with self.assertRaises(FileNotFoundError):
            Runtime(ROOT / "certainly-missing-library")

    def test_disabled_backend_errors(self):
        path = ROOT / "build-nohip-check" / LIB.name
        if not path.is_file():
            self.skipTest("no backend-free build")
        with self.assertRaises(RuntimeError) as caught:
            Runtime(path)
        self.assertEqual(caught.exception.status, 3)
        self.assertEqual(caught.exception.operation, "amdInit")

    def test_fft_disabled_backend(self):
        path = ROOT / "build-nofft-check" / LIB.name
        if not path.is_file():
            self.skipTest("no FFT-disabled build")
        runtime = Runtime(path)
        if not runtime.count():
            self.skipTest("no GPU")
        with Device(runtime=runtime) as device:
            with self.assertRaises(RuntimeError) as caught:
                FFTPlan(device, 8)
            self.assertEqual(caught.exception.status, 3)


@unittest.skipIf(Runtime is None or not LIB.is_file() or not IMAGE.is_file(), "build HIP runtime and kernel fixture")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime(LIB)
        cls.index = None
        for i in range(cls.runtime.count()):
            with Device(i, runtime=cls.runtime) as dev:
                if dev.properties["gcn_arch"].split(":")[0] == "gfx1101":
                    cls.index = i
                    break
        if cls.index is None:
            raise unittest.SkipTest("gfx1101 fixture requires a matching GPU")

    def device(self):
        return Device(self.index, runtime=self.runtime)

    def test_kernel_file_and_memory(self):
        for memory in (False, True):
            with self.subTest(memory=memory), self.device() as device, Stream(device) as stream, MemoryPool(device) as pool:
                options = {"image": IMAGE.read_bytes()} if memory else {"path": IMAGE}
                with Module(device, **options) as module, Event(device) as start, Event(device) as stop:
                    n = 257
                    a = array.array("f", (i * .25 for i in range(n)))
                    b = array.array("f", (i % 7 - 3 for i in range(n)))
                    out = array.array("f", [-999] * n)
                    expected = [x + y for x, y in zip(a, b)]
                    with pool.buffer(len(a)*4, stream) as da, pool.buffer(len(b)*4, stream) as db, pool.buffer(len(out)*4, stream) as dc, pool.buffer(len(out)*4, stream) as copy:
                        da.copy_from_host_async(a)
                        db.copy_from_host_async(b)
                        del a, b
                        gc.collect()  # host exporters must stay alive until synchronization
                        start.record(stream)
                        kernel = module.get_function("vector_add")
                        module.launch(kernel, (3,1,1), (128,1,1), stream, [da,db,dc,KernelArg("int32_t",n)])
                        stop.record(stream)
                        stream.wait_event(stop)
                        copy.copy_from_device_async(dc)
                        copy.copy_to_host_async(out)
                        stream.synchronize()
                        self.assertEqual(list(out), expected)
                        self.assertTrue(stream.ready())
                        self.assertGreaterEqual(start.elapsed_ms(stop), 0)

    def test_resource_order_and_bounds(self):
        with self.device() as device, Stream(device) as stream, MemoryPool(device) as pool:
            with pool.buffer(16, stream) as buffer:
                with self.assertRaises(RuntimeError):
                    pool.close()
                with self.assertRaises(RuntimeError):
                    stream.close()
                with self.assertRaises(ValueError):
                    buffer.copy_from_host_async(bytes(17))
                with self.assertRaises((TypeError, BufferError)):
                    buffer.copy_to_host_async(bytes(16))
                address = buffer.address
            with self.assertRaises(ValueError):
                pool.free(address, stream)
            with self.assertRaises(RuntimeError):
                buffer.copy_from_host_async(bytes(4))
        stream.close()  # idempotent
        with self.assertRaises(RuntimeError):
            stream.synchronize()

    def test_modules_reject_invalid_inputs(self):
        with self.device() as device, Stream(device) as stream:
            with self.assertRaises(RuntimeError):
                Module(device, image=b"not an ELF")
            with Module(device, IMAGE) as module:
                with self.assertRaises(RuntimeError):
                    module.get_function("not_a_symbol")
                kernel = module.get_function("vector_add")
                with self.assertRaises(ValueError):
                    module.launch(kernel, (0,1,1), (1,1,1), stream, [])
                with self.assertRaises(TypeError):
                    module.launch(kernel, (1,1,1), (1,1,1), stream, [42])
            with self.assertRaises(RuntimeError):
                module.launch(kernel, (1,1,1), (1,1,1), stream, [])

    def test_mixed_precision_batched_gemm(self):
        for dtype in ("f32", "f16", "bf16"):
            for ta, tb in ((False,False),(True,False),(False,True),(True,True)):
                with self.subTest(dtype=dtype, ta=ta, tb=tb), self.device() as device, Stream(device) as stream, MemoryPool(device) as pool, Blas(device) as blas:
                    m,n,k,batches = 3,2,4,2
                    lda,ldb,ldc = (k if ta else m)+1,(n if tb else k)+1,m+1
                    sa,sb,sc = lda*(m if ta else k)+2,ldb*(k if tb else n)+2,ldc*n+2
                    a,b,c = [0.0]*(sa*batches),[0.0]*(sb*batches),[-91.0]*(sc*batches)
                    expected = c.copy()
                    for batch in range(batches):
                        for row in range(m):
                            for p in range(k):
                                a[batch*sa+(p+row*lda if ta else row+p*lda)] = (row-p+batch)*.25
                        for col in range(n):
                            for p in range(k):
                                b[batch*sb+(col+p*ldb if tb else p+col*ldb)] = (p+col-batch)*.5
                        for row in range(m):
                            for col in range(n):
                                at=batch*sc+row+col*ldc
                                c[at]=2.0
                                expected[at]=1.25*sum(
                                    a[batch*sa+(p+row*lda if ta else row+p*lda)]*
                                    b[batch*sb+(col+p*ldb if tb else p+col*ldb)]
                                    for p in range(k)) + 1.0
                    def encode(values):
                        if dtype=="bf16":
                            return b"".join(struct.pack("<f",v)[2:] for v in values)
                        return struct.pack("<"+("f" if dtype=="f32" else "e")*len(values),*values)
                    aa,bb = encode(a),encode(b)
                    cc=array.array("f",c)
                    with pool.buffer(len(aa),stream) as da,pool.buffer(len(bb),stream) as db,pool.buffer(len(cc)*4,stream) as dc:
                        da.copy_from_host_async(aa); db.copy_from_host_async(bb); dc.copy_from_host_async(cc)
                        blas.gemm(da,db,dc,m=m,n=n,k=k,input_type=dtype,trans_a=ta,trans_b=tb,
                                  alpha=1.25,beta=.5,lda=lda,ldb=ldb,ldc=ldc,batches=batches,
                                  stride_a=sa,stride_b=sb,stride_c=sc)
                        dc.copy_to_host_async(cc);stream.synchronize()
                        for actual,wanted in zip(cc,expected):
                            self.assertAlmostEqual(actual,wanted,places=5)
                        with self.assertRaises(ValueError):
                            blas.gemm(da,db,dc,m=100,n=100,k=100)
                        with self.assertRaises(RuntimeError):
                            blas.gemm(da,db,dc,m=m,n=n,k=k,input_type=dtype,lda=1)

    def test_fft_forward_inverse_batched(self):
        for n in (7,8):
            with self.subTest(n=n), self.device() as device, Stream(device) as stream, MemoryPool(device) as pool, FFTPlan(device,n,2) as plan:
                values=[complex(i*.25,i%3-1) for i in range(2*n)]
                host=array.array("f",(component for v in values for component in (v.real,v.imag)))
                out=array.array("f",[0.0]*len(host))
                with pool.buffer(len(host)*4,stream) as src,pool.buffer(len(host)*4,stream) as dst:
                    src.copy_from_host_async(host)
                    plan.execute(src,dst)
                    dst.copy_to_host_async(out);stream.synchronize()
                    expected=[sum(values[batch*n+j]*cmath.exp(-2j*math.pi*k*j/n) for j in range(n))
                              for batch in range(2) for k in range(n)]
                    actual=[complex(out[i],out[i+1]) for i in range(0,len(out),2)]
                    for x,y in zip(actual,expected):
                        self.assertLess(abs(x-y),1e-4)
                    plan.execute(dst,dst,inverse=True)  # in-place inverse
                    dst.copy_to_host_async(out);stream.synchronize()
                    for x,y in zip(out,host):
                        self.assertAlmostEqual(x/n,y,places=5)


if __name__ == "__main__":
    unittest.main()

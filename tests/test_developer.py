import importlib.util
import builtins
from pathlib import Path
import unittest
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor

if importlib.util.find_spec("cffi"):
    from compat.native import Device, Graph, Runtime, RuntimeError, Stream, ffi
    from compat.neural import Session
else:
    Runtime=None
ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/"build-hip-check"/"compatcuda.dll"


@unittest.skipUnless(Runtime and LIB.is_file(), "build the Windows HIP runtime")
class DeveloperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime=Runtime(LIB)
        if not cls.runtime.count():raise unittest.SkipTest("no HIP device")

    def session(self):
        return Session(runtime=self.runtime)

    def test_cli_missing_library(self):
        from compat.cli import main
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            status=main(["doctor","--native","--library",str(ROOT/"missing.dll")])
        self.assertEqual(status,1)
        self.assertIn("Local runtime check failed",output.getvalue())

    def test_neural_demo(self):
        from compat.local import demo
        report=demo("softmax",LIB)
        self.assertTrue(report["passed"])
        self.assertEqual(report["elements_checked"],70)

    def test_real_backend_initialization_and_toolchain_report(self):
        from compat.local import initialize
        from compat.cli import _toolchain_report
        report = initialize(LIB)
        self.assertTrue(report["initialized"])
        self.assertEqual(report["backend"], "HIP/ROCm native AMD")
        self.assertEqual(report["device"]["gcn_arch"].split(":")[0], "gfx1101")
        self.assertTrue(_toolchain_report()["ready"])

    def test_raw_handle_validation(self):
        lib=self.runtime.lib
        self.assertEqual(lib.amdDeviceDestroy(ffi.cast("AmdDevice",1)),5)
        self.assertEqual(lib.amdStreamQuery(ffi.cast("AmdStream",1)),5)
        for symbol,kind in (("amdEventDestroy","AmdEvent"),("amdModuleUnload","AmdModule"),
                            ("amdMemPoolDestroy","AmdMemPool"),("amdBlasDestroy","AmdBlasHandle"),
                            ("amdFftDestroy","AmdFftPlan"),("amdGraphDestroy","AmdGraph")):
            self.assertEqual(getattr(lib,symbol)(ffi.cast(kind,1)),5,symbol)
        with Device(runtime=self.runtime) as device:
            self.assertEqual(lib.amdStreamQuery(ffi.cast("AmdStream",device.handle)),5)
            stream=Stream(device);old=stream.handle;stream.close()
            self.assertEqual(lib.amdStreamQuery(old),5)
            for _ in range(20):
                with Stream(device) as fresh:
                    self.assertNotEqual(old,fresh.handle)
            self.assertEqual(lib.amdStreamDestroy(old),5)
            out=ffi.new("AmdStream*")
            self.runtime.check(lib.amdStreamCreate(device.handle,out),"create")
            with ThreadPoolExecutor(2) as workers:
                statuses=list(workers.map(lambda _:lib.amdStreamDestroy(out[0]),range(2)))
            self.assertEqual(sorted(statuses),[0,5])

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"requires NumPy")
    def test_neural_numpy(self):
        import numpy as np
        rng=np.random.default_rng(42)
        with self.session() as session:
            for shape in ((1,2),(3,10),(2,3,128)):
                x=rng.normal(size=shape).astype("float32")
                w=rng.normal(size=shape[-1]).astype("float32")
                value=rng.normal(size=shape).astype("float32")
                np.testing.assert_allclose(session.numpy("rmsnorm",x,w),
                    x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*w,rtol=3e-5,atol=2e-6)
                ex=np.exp(x-x.max(axis=-1,keepdims=True))
                np.testing.assert_allclose(session.numpy("softmax",x),ex/ex.sum(axis=-1,keepdims=True),rtol=3e-5,atol=2e-6)
                np.testing.assert_allclose(session.numpy("swiglu",x,value),x/(1+np.exp(-x))*value,rtol=3e-5,atol=2e-6)
                angles=rng.normal(size=shape[:-1]+(shape[-1]//2,)).astype("float32")
                cosine=np.cos(angles);sine=np.sin(angles)
                expected=np.empty_like(x)
                expected[...,::2]=x[...,::2]*cosine-x[...,1::2]*sine
                expected[...,1::2]=x[...,::2]*sine+x[...,1::2]*cosine
                np.testing.assert_allclose(session.numpy("rope",x,cosine,sine),expected,rtol=3e-5,atol=2e-6)
            with self.assertRaises(TypeError):session.numpy("softmax",x.astype("float64"))
            with self.assertRaises(ValueError):session.numpy("rmsnorm",x,w,epsilon=-1)
            with self.assertRaises(ValueError):session.numpy("softmax",np.array([[float("nan")]],dtype="float32"))

    @unittest.skipUnless(importlib.util.find_spec("numpy"),"requires NumPy")
    def test_graph_replay_and_pinning(self):
        import numpy as np
        with self.session() as session:
            x=np.arange(30,dtype="float32").reshape(3,10)/10
            weights=np.ones(10,dtype="float32")
            with session.pool.buffer(x.nbytes,session.stream) as a,session.pool.buffer(weights.nbytes,session.stream) as w,session.pool.buffer(x.nbytes,session.stream) as b,session.pool.buffer(x.nbytes,session.stream) as c:
                a.copy_from_host_async(x);w.copy_from_host_async(weights);session.stream.synchronize()
                graph=Graph(session.stream)
                try:
                    with graph:
                        session.ops.rmsnorm(a,w,b,3,10)
                        session.ops.softmax(b,c,3,10)
                    with self.assertRaises(RuntimeError):a.close()
                    with self.assertRaises(RuntimeError):session.ops.close()
                    for offset in (0,1,2):
                        input_data=x+offset
                        a.copy_from_host_async(input_data)
                        graph.launch()
                        output=np.empty_like(x);c.copy_to_host_async(output);session.stream.synchronize()
                        expected=input_data/np.sqrt(np.mean(input_data**2,axis=-1,keepdims=True)+1e-5)
                        expected=np.exp(expected-expected.max(axis=-1,keepdims=True));expected/=expected.sum(axis=-1,keepdims=True)
                        np.testing.assert_allclose(output,expected,rtol=3e-5,atol=2e-6)
                    old=graph.handle
                finally:graph.close()
                self.assertEqual(self.runtime.lib.amdGraphLaunch(old,session.stream.handle),5)
                with self.assertRaises(ValueError):
                    with Graph(session.stream):
                        raise ValueError("cancel capture")
                session.stream.synchronize()

    @unittest.skipUnless(importlib.util.find_spec("torch"),"requires PyTorch")
    def test_torch_inference_bridge(self):
        import torch
        with self.session() as session:
            x=torch.linspace(-2,2,30,dtype=torch.float32).reshape(3,10)
            actual=session.torch("softmax",x)
            torch.testing.assert_close(actual,torch.softmax(x,dim=-1),rtol=3e-5,atol=2e-6)
            self.assertEqual(actual.device.type,"cpu")
            with self.assertRaises(TypeError):session.torch("softmax",x.requires_grad_())

    @unittest.skipUnless(importlib.util.find_spec("torch"),"requires PyTorch")
    def test_torch_module_adapter_and_report(self):
        import torch
        from compat.framework import AmdInferenceModule, framework_report
        self.assertFalse(framework_report()["torch_cuda_replacement"])
        x=torch.linspace(-2,2,30,dtype=torch.float32).reshape(3,10)
        with AmdInferenceModule("softmax",runtime=self.runtime) as module:
            self.assertIsInstance(module,torch.nn.Module)
            torch.testing.assert_close(module(x),torch.softmax(x,dim=-1),rtol=3e-5,atol=2e-6)
            with self.assertRaises(TypeError):module(x.requires_grad_())
        with self.assertRaises(builtins.RuntimeError):
            module(x.detach())


if __name__=="__main__":unittest.main()

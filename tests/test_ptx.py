import array
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compat.cli import _find_hipcc
from compat.ptx import PtxError, compile_ptx, lower_ptx

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/vector_add.ptx"
BOUNDED_FIXTURE = ROOT / "examples/vector_add_bounded.ptx"
AFFINE_FIXTURE = ROOT / "examples/affine_u32.ptx"
SAXPY_FIXTURE = ROOT / "examples/saxpy_bounded.ptx"


class PtxTests(unittest.TestCase):
    def test_lowering_and_rejections(self):
        source = FIXTURE.read_text()
        self.assertIn('void ptx_vector_add', lower_ptx(source))
        self.assertIn('__fadd_rn', lower_ptx(source))
        lowered_bounded = lower_ptx(BOUNDED_FIXTURE.read_text())
        self.assertIn('if (r_p0) goto L_ACTIVE;', lowered_bounded)
        self.assertIn('goto L_DONE;', lowered_bounded)
        affine = lower_ptx(AFFINE_FIXTURE.read_text())
        self.assertIn('reinterpret_cast<uint32_t*>', affine)
        self.assertIn('static_cast<uint64_t>', affine)
        saxpy = lower_ptx(SAXPY_FIXTURE.read_text())
        self.assertIn('__fmul_rn', saxpy)
        self.assertIn('__fmaf_rn', saxpy)
        invalid = [source.replace('add.rn.f32', 'atom.global.add.f32'),
                   source.replace('ret;', '@%p ret;'),
                   source.replace('%f0, %f1;', '%f0, %f9;'),
                   source.replace('mov.u32 %r0, %ctaid.x;', 'mov.u32 %r0, %r1;'),
                   source.replace('.address_size 64', '.address_size 32'),
                   source.replace('ret;', 'ret; ret;'),
                   source + 'extern int evil;',
                   source.replace('4;', '18446744073709551616;'),
                   source.replace('%r<4>', '%r<9999999>')]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(PtxError):
                lower_ptx(item)

    def test_rejection_does_not_invoke_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.ptx'
            path.write_text('.version 7.0\n unsupported')
            with patch('compat.ptx.compile_code_object') as backend:
                with self.assertRaises(PtxError):
                    compile_ptx(path, Path(directory)/'out.hsaco', 'gfx1101', compiler='unused')
                backend.assert_not_called()
                self.assertFalse((Path(directory)/'out.hsaco').exists())

    @unittest.skipUnless(_find_hipcc() and (ROOT/'build-hip-check/compatcuda.dll').exists(),
                         'requires Windows HIP SDK and built runtime')
    def test_translated_ptx_executes(self):
        try:
            from compat.native import Runtime, Device, Stream, MemoryPool, Module
        except ImportError:
            self.skipTest('requires native CFFI extra')
        runtime = Runtime(ROOT/'build-hip-check/compatcuda.dll')
        index = None
        for i in range(runtime.count()):
            with Device(i, runtime=runtime) as device:
                if device.properties['gcn_arch'].split(':')[0] == 'gfx1101':
                    index = i
                    break
        if index is None:
            self.skipTest('requires verified gfx1101 target')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'translated.hsaco'
            original = FIXTURE.read_bytes()
            self.assertEqual(compile_ptx(FIXTURE, output, 'gfx1101', compiler=_find_hipcc()), 0)
            self.assertEqual(FIXTURE.read_bytes(), original)
            with self.assertRaises(ValueError):
                compile_ptx(FIXTURE, output, 'gfx1101', compiler=_find_hipcc())
            with Device(index, runtime=runtime) as device, Stream(device) as stream, MemoryPool(device) as pool, Module(device, output) as module:
                for blocks, threads in ((1,32), (3,64)):
                    count = blocks*threads
                    a = array.array('f', (i*.25-12 for i in range(count)))
                    b = array.array('f', (i%7-3 for i in range(count)))
                    result = array.array('f', [-999]*count)
                    with pool.buffer(count*4, stream) as da, pool.buffer(count*4, stream) as db, pool.buffer(count*4, stream) as dc:
                        da.copy_from_host_async(a)
                        db.copy_from_host_async(b)
                        module.launch(module.get_function('ptx_vector_add'), (blocks,1,1), (threads,1,1), stream, [da,db,dc])
                        dc.copy_to_host_async(result)
                        stream.synchronize()
                        self.assertEqual(list(result), [x+y for x,y in zip(a,b)])

    @unittest.skipUnless(_find_hipcc() and (ROOT/'build-hip-check/compatcuda.dll').exists(),
                         'requires Windows HIP SDK and built runtime')
    def test_bounded_ptx_executes_extra_threads(self):
        try:
            from compat.native import Runtime, Device, Stream, MemoryPool, Module, KernelArg
        except ImportError:
            self.skipTest('requires native CFFI extra')
        runtime = Runtime(ROOT/'build-hip-check/compatcuda.dll')
        with Device(0, runtime=runtime) as device:
            if device.properties['gcn_arch'].split(':')[0] != 'gfx1101':
                self.skipTest('requires verified gfx1101 target')
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'bounded.hsaco'
                self.assertEqual(compile_ptx(BOUNDED_FIXTURE, output, 'gfx1101', compiler=_find_hipcc()), 0)
                with Stream(device) as stream, MemoryPool(device) as pool, Module(device, output) as module:
                    count, launched = 179, 192
                    a = array.array('f', (i*.5 for i in range(count)))
                    b = array.array('f', (3-i*.25 for i in range(count)))
                    result = array.array('f', [-1]*count)
                    with pool.buffer(count*4, stream) as da, pool.buffer(count*4, stream) as db, pool.buffer(count*4, stream) as dc:
                        da.copy_from_host_async(a); db.copy_from_host_async(b)
                        module.launch(module.get_function('ptx_vector_add_bounded'), (3,1,1), (64,1,1), stream, [da,db,dc,KernelArg('uint32_t',count)])
                        dc.copy_to_host_async(result); stream.synchronize()
                    self.assertEqual(list(result), [x+y for x,y in zip(a,b)])

    @unittest.skipUnless(_find_hipcc() and (ROOT/'build-hip-check/compatcuda.dll').exists(),
                         'requires Windows HIP SDK and built runtime')
    def test_integer_memory_and_wide_multiply_execute(self):
        try:
            from compat.native import Runtime, Device, Stream, MemoryPool, Module, KernelArg
        except ImportError:
            self.skipTest('requires native CFFI extra')
        runtime = Runtime(ROOT/'build-hip-check/compatcuda.dll')
        with Device(0, runtime=runtime) as device:
            if device.properties['gcn_arch'].split(':')[0] != 'gfx1101':
                self.skipTest('requires verified gfx1101 target')
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'affine.hsaco'
                self.assertEqual(compile_ptx(AFFINE_FIXTURE, output, 'gfx1101', compiler=_find_hipcc()), 0)
                with Stream(device) as stream, MemoryPool(device) as pool, Module(device, output) as module:
                    count = 91
                    source = array.array('I', (i * 101 for i in range(count)))
                    result = array.array('I', [0] * count)
                    with pool.buffer(source.itemsize*count, stream) as si, pool.buffer(result.itemsize*count, stream) as so:
                        si.copy_from_host_async(source)
                        module.launch(module.get_function('ptx_affine_u32'), (2,1,1), (64,1,1), stream, [si,so,KernelArg('uint32_t',count)])
                        so.copy_to_host_async(result); stream.synchronize()
                    self.assertEqual(list(result), [value*3+7 for value in source])

    @unittest.skipUnless(_find_hipcc() and (ROOT/'build-hip-check/compatcuda.dll').exists(),
                         'requires Windows HIP SDK and built runtime')
    def test_float_fma_executes_with_bounds(self):
        try:
            from compat.native import Runtime, Device, Stream, MemoryPool, Module, KernelArg
        except ImportError:
            self.skipTest('requires native CFFI extra')
        runtime = Runtime(ROOT/'build-hip-check/compatcuda.dll')
        with Device(0, runtime=runtime) as device:
            if device.properties['gcn_arch'].split(':')[0] != 'gfx1101':
                self.skipTest('requires verified gfx1101 target')
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'saxpy.hsaco'
                self.assertEqual(compile_ptx(SAXPY_FIXTURE, output, 'gfx1101', compiler=_find_hipcc()), 0)
                with Stream(device) as stream, MemoryPool(device) as pool, Module(device, output) as module:
                    count, alpha = 129, 1.25
                    x = array.array('f', (i*.25-17 for i in range(count)))
                    y = array.array('f', (3-i*.125 for i in range(count)))
                    result = array.array('f', [-99]*count)
                    with pool.buffer(count*4, stream) as dx, pool.buffer(count*4, stream) as dy, pool.buffer(count*4, stream) as dz:
                        dx.copy_from_host_async(x); dy.copy_from_host_async(y)
                        module.launch(module.get_function('ptx_saxpy_bounded'), (3,1,1), (64,1,1), stream, [dx,dy,dz,KernelArg('uint32_t',count),KernelArg('float',alpha)])
                        dz.copy_to_host_async(result); stream.synchronize()
                    for actual, left, right in zip(result, x, y):
                        self.assertAlmostEqual(actual, left*alpha+right, places=5)

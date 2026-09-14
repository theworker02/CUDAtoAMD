import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from compat.compiler import compile_code_object
from compat.cli import _find_hipcc

ROOT=Path(__file__).resolve().parents[1]


class CompilerTests(unittest.TestCase):
    def test_rejects_existing_output_and_wrong_target(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"new.hsaco"
            with self.assertRaises(ValueError):
                compile_code_object(ROOT/"examples/cuda_vector.cu",output,"sm_89",compiler="unused")
            with self.assertRaises(ValueError):
                compile_code_object(ROOT/"examples/cuda_vector.cu",ROOT/"examples/cuda_vector.cu","gfx1101",compiler="unused")

    @unittest.skipUnless(_find_hipcc() and (ROOT/"build-hip-check/compatcuda_contract.exe").is_file(),"build HIP CUDA contract test")
    def test_cuda_source_to_real_application(self):
        compiler=_find_hipcc()
        source=ROOT/"examples/cuda_vector.cu"
        before=source.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"vector.hsaco"
            self.assertEqual(compile_code_object(source,output,"gfx1101",compiler=compiler),0)
            env=dict(os.environ)
            env["PATH"]=str(Path(compiler).parent)+os.pathsep+env.get("PATH","")
            subprocess.run([str(ROOT/"build-hip-check/compatcuda_contract.exe"),str(output)],env=env,check=True)
            self.assertEqual(source.read_bytes(),before)
            with self.assertRaises(ValueError):
                compile_code_object(source,output,"gfx1101",compiler=compiler)

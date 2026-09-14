import tempfile
import unittest
from pathlib import Path

from compat.analyzer import analyze


class AnalyzerTests(unittest.TestCase):
    def test_detects_api_headers_and_nvidia_specific_constructs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "kernel.cu").write_text('#include <cuda_runtime.h>\nvoid f(){ cudaMalloc(0, 0); k<<<1, 1>>>(); asm("bar"); }', encoding="utf-8")
            report = analyze(root)
        self.assertEqual(report["summary"]["by_status"]["DIRECT"], 1)
        self.assertIn("cuda_runtime.h", report["cuda_headers"])
        self.assertEqual(report["summary"]["special_constructs"]["kernel_syntax"], 1)
        self.assertEqual(report["summary"]["special_constructs"]["inline_ptx"], 1)


if __name__ == "__main__":
    unittest.main()

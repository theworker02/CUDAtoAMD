import tempfile
import unittest
from pathlib import Path

from compat.analyzer import analyze


class BinaryAnalysisTests(unittest.TestCase):
    def test_identifies_ptx_as_translation_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "kernel.ptx").write_text(".version 8.0", encoding="utf-8")
            report = analyze(root)
        finding = report["findings"][0]
        self.assertEqual(finding["binary_format"], "PTX text")
        self.assertEqual(finding["assessment"], "requires translation")


if __name__ == "__main__":
    unittest.main()

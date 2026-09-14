import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from compat.cli import main
from compat.wheel import inspect_wheel, wheel_launchable


def make_wheel(path, files, requires=()):
    metadata = "Metadata-Version: 2.1\nName: fixture\nVersion: 0\n" + "".join(f"Requires-Dist: {item}\n" for item in requires)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fixture-0.dist-info/METADATA", metadata)
        for name, data in files.items(): archive.writestr(name, data)


class WheelTests(unittest.TestCase):
    def test_blocked_cuda_native_wheel_is_reported_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.whl"
            make_wheel(path, {"fixture/backend.pyd": b"MZ...cudart64...cublas64"}, ["nvidia-cuda-runtime-cu12"])
            report = inspect_wheel(path)
        self.assertEqual(report["verdict"], "blocked")
        self.assertFalse(wheel_launchable(report))
        self.assertEqual(len(report["cuda_importing_extensions"]), 1)
        self.assertEqual(len(report["sha256"]), 64)

    def test_python_only_wheel_and_cli_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pure.whl"
            make_wheel(path, {"fixture/__init__.py": b"x=1"})
            self.assertTrue(wheel_launchable(inspect_wheel(path)))
            output = io.StringIO()
            with contextlib.redirect_stdout(output): status = main(["wheel-doctor", str(path), "--format", "json"])
        self.assertEqual(status, 0)
        self.assertIn('"verdict": "python_only_or_no_cuda_binary_detected"', output.getvalue())

    def test_malformed_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.whl"; path.write_bytes(b"not a zip")
            with self.assertRaises(ValueError): inspect_wheel(path)

    def test_python_launcher_refuses_blocked_wheel_without_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel, script = root / "cuda.whl", root / "app.py"
            make_wheel(wheel, {"fixture/backend.pyd": b"cudart64"})
            script.write_text("raise RuntimeError('must not run')")
            output = io.StringIO()
            with contextlib.redirect_stdout(output): status = main(["python", "--wheel", str(wheel), str(script)])
        self.assertEqual(status, 2)
        self.assertIn("blocked launch", output.getvalue())

    def test_python_launcher_requires_exact_reviewed_wheel_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel, script = root / "pure.whl", root / "app.py"
            make_wheel(wheel, {"fixture/__init__.py": b"x=1"})
            script.write_text("raise RuntimeError('must not run')")
            output = io.StringIO()
            with contextlib.redirect_stdout(output): status = main(["python", "--wheel", str(wheel), "--wheel-sha256", "0" * 64, str(script)])
        self.assertEqual(status, 2)
        self.assertIn("SHA-256", output.getvalue())

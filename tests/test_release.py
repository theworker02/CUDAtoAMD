from pathlib import Path
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

from compat.cli import main
from compat.release import release_ready, release_report


class ReleaseTests(unittest.TestCase):
    def test_release_metadata_is_consistent(self):
        report = release_report()
        self.assertTrue(release_ready(report), report)
        self.assertEqual(report["version"], "1.4.0")

    def test_release_cli_reports_missing_requested_library(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["release-check", "--library", str(Path("missing-compatcuda.dll")), "--format", "json"])
        self.assertEqual(result, 1)
        self.assertIn('"native_library_present": false', output.getvalue())

    def test_installed_package_mode_does_not_require_source_tree(self):
        import compat.release as release
        with tempfile.TemporaryDirectory() as directory, patch.object(release, "ROOT", Path(directory)):
            report = release.release_report()
        self.assertFalse(report["source_checkout"])
        self.assertFalse(release.release_ready(report))
        self.assertIn("source checkout", report["message"])

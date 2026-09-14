from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class ProjectSiteTests(unittest.TestCase):
    def test_project_logo_and_static_site_are_present(self):
        logo = ROOT / "assets" / "cudatoamd-logo.svg"
        site = ROOT / "docs" / "site"
        self.assertTrue(logo.is_file())
        self.assertIn("CUDAtoAMD", logo.read_text(encoding="utf-8"))
        for path in ("index.html", "styles.css", "app.js", "404.html", "assets/cudatoamd-logo.svg"):
            self.assertTrue((site / path).is_file(), path)

    def test_site_declares_its_local_assets_and_honest_scope(self):
        page = (ROOT / "docs" / "site" / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="styles.css"', page)
        self.assertIn('src="app.js"', page)
        self.assertIn("assets/cudatoamd-logo.svg", page)
        self.assertIn("does <strong>not</strong> run arbitrary CUDA binaries", page)

    def test_pages_workflow_uploads_only_the_static_site(self):
        workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
        self.assertIn("actions/upload-pages-artifact@v3", workflow)
        self.assertIn("path: docs/site", workflow)
        self.assertIn("actions/deploy-pages@v4", workflow)

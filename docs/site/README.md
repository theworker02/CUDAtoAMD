# GitHub Pages site

`docs/site/` is a dependency-free static project site. It is intentionally self-contained: the GitHub Pages workflow uploads this directory only, so it does not require a JavaScript build, external fonts, analytics, cookies, or a hosted backend.

## Publish after creating the GitHub repository

1. Create a GitHub repository, or add the appropriate remote to this checkout.
2. Commit this project and push the `main` branch.
3. In the repository's **Settings → Pages**, select **GitHub Actions** as the build and deployment source.
4. The `Deploy GitHub Pages` workflow deploys on the next `main` push that changes this directory, or can be run manually from the Actions tab.

GitHub supplies the final site URL after the first successful deployment. The workflow makes no release, package publication, or external service change beyond the GitHub Pages deployment that GitHub Actions performs after it has been enabled.

For local review, open `index.html` in a browser. The copy-command control may need a secure origin (such as the deployed Pages URL) before browser clipboard access is available.

# Publishing `langchain-keenable` to PyPI

This package publishes to PyPI via **Trusted Publishing (OIDC)** — GitHub Actions
mints a short-lived identity token and PyPI trusts it, so **no API tokens or
passwords are stored anywhere**. The workflow is
[`.github/workflows/publish.yml`](.github/workflows/publish.yml); it runs when a
GitHub **Release** is published.

## One-time setup (do this once, by a PyPI account owner)

The PyPI account needs a **verified email** and **2FA enabled** first
(https://pypi.org/manage/account/). Then:

1. Go to **https://pypi.org/manage/account/publishing/** ("Publishing" →
   "Add a new pending publisher"). A *pending* publisher lets you wire this up
   **before the project exists** — the first release creates the project.
2. Fill in exactly:
   - **PyPI Project Name:** `langchain-keenable`
   - **Owner:** `keenableai`
   - **Repository name:** `langchain-keenable`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. Save. (Values must match the workflow exactly, including the environment name
   `pypi` — a mismatch is the #1 reason publishing silently fails with
   "not a trusted publisher".)
4. *(Recommended)* In this GitHub repo → **Settings → Environments → New
   environment → `pypi`**, and optionally add required reviewers so a release
   needs a human approval before it publishes.

That's it — no secrets to add to GitHub.

## Cutting a release

1. Bump `version` in `pyproject.toml` (PyPI never lets you re-upload the same
   version). Commit + push to `main`.
2. Tag and create a GitHub Release:
   ```bash
   git tag v0.1.0           # tag must match pyproject version
   git push origin v0.1.0
   gh release create v0.1.0 --title "v0.1.0" --notes "Initial release"
   ```
   (Or use the GitHub UI: Releases → Draft a new release.)
3. The **Publish to PyPI** workflow runs: build → `twine check` → publish via
   OIDC. Watch it under the repo's **Actions** tab. On success the package is
   live at https://pypi.org/project/langchain-keenable/.
4. After the first successful publish you can tighten the PyPI publisher to be
   project-scoped (it already is, by project name).

You can also trigger a (re)publish manually from **Actions → Publish to PyPI →
Run workflow** (`workflow_dispatch`) — useful once a tag/release already exists.

## Pre-release checks (run locally first)

```bash
rm -rf dist && uv build
uvx twine check dist/*
uv run --group test pytest tests/unit_tests   # 59 passing, offline
```

## Troubleshooting ("we tried and it was silent")

- **No verification email on signup:** check Spam; resend from
  https://pypi.org/manage/account/ → Account emails; the link expires — request a
  fresh one. Corporate mail (`@keenable.ai`) may filter it; try a personal email
  and add the work address later.
- **2FA not enabled:** you cannot create tokens *or* publish without it. Enable a
  TOTP app under account settings.
- **Workflow runs but PyPI rejects with "not a trusted publisher":** the pending
  publisher fields don't match — most often the **environment name** (`pypi`) or
  the **workflow filename** (`publish.yml`). Re-check step 2 above.
- **`Missing id-token: write`:** the `publish` job must keep
  `permissions: id-token: write` (it does). Don't remove it.
- **Re-uploading the same version:** PyPI refuses silently-looking 400s — bump
  the version; you can't overwrite `0.1.0` once published.

## Manual fallback (token, only if Actions is unavailable)

Trusted Publishing is preferred. If you must publish from a laptop:
```bash
rm -rf dist && uv build
UV_PUBLISH_TOKEN=pypi-XXXX uv publish     # account- or project-scoped token; user is __token__
```
Create the token at https://pypi.org/manage/account/token/ (first one is
account-scoped because the project doesn't exist yet; switch to a
project-scoped token after the first publish). Prefer the OIDC workflow above.

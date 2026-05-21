# Version Management Guide

## Overview

This project uses **[python-semantic-release](https://python-semantic-release.readthedocs.io/)** to manage versions from [Conventional Commits](https://www.conventionalcommits.org/) on the `main` branch.

**Key principles:**

- **Production versions**: CI bumps `biolmai/__version__` and creates a Git tag (e.g. `v0.5.0`) on `main`
- **PyPI publish**: Triggered when a GitHub Release is published (see `.github/workflows/publish.yml`)
- **Non-conventional commits**: Ignored for versioning (no CI failure)
- **TestPyPI**: Use `-rc` suffixes manually when needed (e.g. `0.5.0-rc.1`)

## How It Works

1. **Commit with conventional format** (e.g. `feat:`, `fix:`, `BREAKING:`) or squash-merge a PR whose title includes them (e.g. `PD-52 feat: ...`)
2. **Push to `main`**
3. **CI runs python-semantic-release** which:
   - Analyzes commit messages since the last tag
   - Determines version bump (major/minor/patch)
   - Updates `biolmai/__init__.py` and `pyproject.toml` (`project.version`)
   - Creates git tag (e.g. `v0.5.0`)
   - Opens a GitHub Release
   - Commits the version bump with `[skip ci]` to avoid loops
4. **Publish workflow** uploads to PyPI when the GitHub Release is published

## Commit Message Format

| Commit Type | Version Bump | Example |
|------------|--------------|---------|
| `feat: ...` | **Minor** (0.4.0 → 0.5.0) | `feat: add protocol batch API` |
| `fix: ...` | **Patch** (0.4.0 → 0.4.1) | `fix: resolve auth timeout` |
| `BREAKING: ...` | **Major** (0.4.0 → 1.0.0) | `BREAKING: change CLI entry point` |
| Other / no prefix | **None** | `update readme`, `PD-52 wip` |

Ticket prefixes are fine when combined with a conventional type: `PD-52 feat: add versioning` or `feat(PD-52): add versioning`.

## What NOT to Do

**Do not manually edit `__version__` in `biolmai/__init__.py` for production releases.** That causes mismatches with git tags and PyPI.

```bash
# BAD — do not do this for production
vim biolmai/__init__.py  # hand-edit __version__
```

## Correct Workflow

### Regular releases

1. Make changes and merge to `main` with a conventional commit (or squash PR title).
2. Wait for CI: tests, then the `version` job on `main`.
3. Confirm the new tag and GitHub Release (e.g. `v0.5.0`).
4. The **Publish to PyPI** workflow runs on release publish and uploads the package.

### Docs on GitHub Pages

Docs still deploy from the **`production`** branch (`.github/workflows/docs.yml`). After a release on `main`, merge `main` → `production` when documentation should go live.

### Local testing

You do not need to bump the version for local testing. Build and test at the current version:

```bash
pip install -e .
pytest -q tests/
```

### TestPyPI (release candidates)

For pre-release uploads without conflicting with the next production version:

```bash
# Bump to RC locally (do not tag for production)
# Edit biolmai/__init__.py to e.g. 0.5.0-rc.1 only for a TestPyPI trial
make dist
make testrelease
```

Or use **workflow_dispatch** on the Publish workflow (TestPyPI) after configuring `TESTPYPI_API_TOKEN`.

## Configuration

See `[tool.semantic_release]` in `pyproject.toml`. The canonical version lives in `biolmai/__init__.py` (also written to `pyproject.toml` by CI). `setup.py` reads `__init__.py`; `docs/conf.py` imports `biolmai`.

## Troubleshooting

```bash
# Current package version
python -c "import biolmai; print(biolmai.__version__)"

# Latest tag
git describe --tags --abbrev=0

# Dry-run (requires python-semantic-release installed)
semantic-release version --print
```

If `package` version and git tags disagree, align with the latest tag and let the next conventional commit on `main` produce a correct release.

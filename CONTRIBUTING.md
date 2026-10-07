# Contributing Guidelines

Thank you for contributing to the Anonymisation Repositories suite!

## Branching & Commit Policy

We enforce **GitHub Flow**:
- Default branch is `main` (protected: changes land through a pull request, squash merge only).
- Create short-lived branches following kebab-case naming. GitHub rejects any other name:
  - `feat/<short-kebab-description>`
  - `fix/<short-kebab-description>`
  - `refactor/<short-kebab-description>`
  - `docs/<short-kebab-description>`
  - `chore/<short-kebab-description>`
  - `test/`, `perf/` and `ci/` are also accepted.

### Prohibited Branch Names
- No uppercase letters or underscores.
- No version numbers in branch names (use Git tags `vX.Y.Z` for releases).
- No status words like `final`, `latest`, `updated`, or `testing`.

## Commit Messages

Follow Conventional Commits:
`feat: add pixelate mode for human masking`
`fix: map plate boxes back to original resolution`

## Development Workflow

1. Install the locked dependencies and the package in editable mode:
   ```bash
   pip install -r requirements-dev.lock
   pip install --no-deps -e .
   pre-commit install
   ```
2. Run tests and linting:
   ```bash
   pytest
   ruff check .
   black --check .
   ```
3. Submit a Pull Request targeting `main`.

## Dependencies

Direct dependencies are declared in `pyproject.toml` with a floor and an upper bound.
After changing them, regenerate both lock files:

```bash
pip-compile --allow-unsafe --strip-extras --no-emit-trusted-host \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -o requirements.lock pyproject.toml
pip-compile --allow-unsafe --strip-extras --no-emit-trusted-host \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra dev -o requirements-dev.lock pyproject.toml
```

## Releasing

A merged version bump is a release; nobody pushes tags or retags images by hand.

1. In your PR, bump `__version__` in `src/image_deidentification/__init__.py` (SemVer:
   breaking change → major, new feature → minor, fix → patch). `pyproject.toml` reads the
   version from there.
2. In `CHANGELOG.md`, move the `[Unreleased]` entries under `## [X.Y.Z] - YYYY-MM-DD` and update
   the compare links. CI fails the PR if the new version has no CHANGELOG section.
3. After review, squash merge. Once the tests pass on `main`, CI pushes
   `ghcr.io/datakaveri/skald-image:vX.Y.Z` and `:latest`, creates the `vX.Y.Z` tag and the
   GitHub Release. `:latest` only moves forward: a lower version (e.g. a hotfix for an older
   line) gets its tag but leaves `:latest` where it is.

PRs that don't bump the version publish only `:main` and `:sha-<commit>`.

If a release run fails after the tag exists (tags cannot be changed or re-pushed), re-run it:

```bash
gh workflow run release.yml -f tag=vX.Y.Z
```

## Data and Model Weights

- Never commit images, sample data or model weights. `.gitignore` excludes `*.pt` and
  `models/`; weights are fetched by `scripts/download_models.py`.
- The pre-commit hooks reject files over 5 MB and scan for secrets.

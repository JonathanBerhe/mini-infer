# ADR-001: uv as package manager

Date: 2026-04-25
Status: Accepted

## Context

mini-infer needs a Python package manager with lockfiles and reproducible
installs. The dependencies are heavy (PyTorch, Transformers). CI must be
fast, and a clean clone must be easy to set up.

## Decision

Use `uv` (astral-sh) as the package manager and venv tool. It is a fast Rust
resolver and installer with a native lockfile. `uv.lock` is in git, so every
clone gets the same dependency graph.

## Alternatives Considered

- pip + venv + pip-tools: three tools and a manual compile step for a
  lockfile.
- Poetry: a slow resolver on heavy ML stacks, and a nonstandard lockfile.
- conda or mamba: a heavier install, and a split between PyPI and
  conda-forge.

## Consequences

- One tool replaces three, and `uv sync` sets up the environment.
- uv is younger than pip and Poetry, so edge-case bugs are more likely.
- A switch is cheap: `pyproject.toml` follows PEP 621, and `uv export`
  writes a `requirements.txt`.

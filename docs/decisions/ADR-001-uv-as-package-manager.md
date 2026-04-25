# ADR-001: uv as package manager

Date: 2026-04-25
Status: Accepted

## Context

mini-infer needs a Python package manager that handles dependency resolution, virtual environments, lockfiles, and reproducible installs. The project pulls in heavy ML dependencies (PyTorch, Transformers) plus a stack of dev tools (ruff, mypy, pytest), and is expected to add CUDA-specific extras as it grows. The choice should make CI fast, local setup painless on a clean clone, and migration off the tool reasonable if priorities change later.

## Decision

Use `uv` (astral-sh) as the package manager and venv tool.

`uv.lock` is committed to git so any clone reproduces the same dependency graph.

## Alternatives Considered

- **pip + venv (+ pip-tools)**: ubiquitous and shipped with the standard library, but requires three tools (`venv`, `pip`, `pip-tools`) and a hand-rolled compile step to get a real lockfile. More moving parts, slower installs as the dep set grows.
- **Poetry**: mature, popular, integrated venv + lock + publish workflow. Downsides: slow resolver on heavy ML stacks, occasional friction with PyTorch wheels, lockfile format is non-standard, pyproject layout was historically not PEP 621 compliant (1.5+ closed the gap).
- **conda / mamba**: strong fit for scientific stacks where CUDA, MKL, and cuDNN need to be coordinated. Downsides: heavier install footprint, ecosystem split between PyPI and conda-forge, slower iteration loop on pure-Python deps.
- **uv**: Rust binary, very fast resolver and installer, native lockfile (`uv.lock`), PEP 621 / PEP 735 compliant `pyproject.toml`, drop-in venv management, broad PyPI compatibility.

## Consequences

- **Positive**: faster installs (matters as deps accumulate); one tool instead of three; lockfile from day one; simple onboarding (`uv sync` and the env is ready).
- **Negative**: uv is younger than pip and Poetry, so the community is smaller and edge-case bugs are likelier than with established tools.
- **Reversibility**: low cost. uv reads PEP 621 `pyproject.toml`, so project metadata stays portable. The dev `[dependency-groups]` block can be migrated to a Poetry or pip-tools equivalent. `uv.lock` can be exported to `requirements.txt` via `uv export` if we ever need to switch.

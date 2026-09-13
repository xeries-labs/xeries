# Implementation Plan: Docs and packaging cleanup

**Branch**: `014-docs-packaging-cleanup` | **Date**: 2026-09-14 | **Spec**: [spec.md](./spec.md)  
**Home repo**: xeries  
**Input**: Feature specification from `/specs/014-docs-packaging-cleanup/spec.md`

## Summary

Hygiene-only change: fix MkDocs navigation and API documentation, render `examples/` notebooks on the docs site with saved outputs, and straighten Hatch/uv extras and lockfile handling. No public symbols under `src/xeries/` are added, removed, or renamed.

## Technical Context

**Language/Version**: Python ≥3.10 (`pyproject.toml`).  
**Primary Dependencies**: Existing runtime (`scikit-learn`, `pandas`, `shap`, `numpy`, `joblib`); optional `skforecast`; docs stack (`mkdocs`, `mkdocs-material`, `mkdocs-jupyter`, `mkdocstrings`); notebook extra (`jupyter`, `matplotlib`, `lightgbm`).  
**Storage**: N/A (docs and packaging metadata).  
**Testing**: `pytest` notebook structure tests; `mkdocs build --strict`; `uv build` wheel inspection.  
**Target Platform**: Library docs (GitHub Pages) and PyPI wheel.  
**Project Type**: Single-project library (`src/xeries/`).  
**Performance Goals**: Docs build must complete in CI without executing notebooks at build time.  
**Constraints**: Constitution repo-scope: `lightgbm` only as optional extra. Do not edit `.specify/`.  
**Scale/Scope**: Docs, `pyproject.toml`, examples notebooks, CI workflows, citation metadata. No `src/` API work.

## Constitution Check

| Principle | Gate question | Verdict (Pass / Needs justification / N/A) | Notes |
| --- | --- | --- | --- |
| I. Specs before code | Does this feature have a `spec.md` authored before any implementation commit? For Backfilled items, is the "Backfilled from implementation" banner present? | Pass | `spec.md` in this folder. Working-tree drafts of docs/pyproject from an earlier session are completed under this spec, not committed as an unspecialized change. |
| II. Agent-agnosticism | Does this plan avoid introducing agent-specific constructs (Copilot-only / Cursor-only prompts) in `src/`, `tests/`, or `docs/`? Shim-only additions are fine and live in per-agent directories of the consumer repo. | Pass | MkDocs hook is generic Python. |
| III. Test-first for new contracts | Is there at least one failing contract test planned (or present, for backfills) for every new public symbol? | N/A | No new public symbols. Notebook discovery tests are updated to match existing notebooks. |
| IV. Typed public surface | Are all new public symbols typed? Will `ty check src` and `ruff` pass? | N/A | No new public symbols. `docs/hooks.py` is not the library surface. |
| V. Reproducibility discipline (bench only) | If Home repo is `xeries-bench`: … | N/A | Home repo is `xeries`. |
| Repo scope | Does this plan respect the Repo-scope table? Specifically: are `lightgbm`, `shapiq`, notebook-heavy artefacts kept out of `xeries` runtime deps and confined to `xeries-bench` (or to `xeries`'s optional extras)? | Pass | `lightgbm` moves to `[project.optional-dependencies] notebooks` only. Notebooks stay in `examples/` (library tutorials), not a new runtime dep. |

## Project Structure

### Documentation (this feature)

```text
specs/014-docs-packaging-cleanup/
├── spec.md
├── plan.md
└── tasks.md
```

### Source Code (repository root)

```text
pyproject.toml
uv.lock
.gitignore
mkdocs.yml
docs/
├── hooks.py
├── examples.md
├── index.md
├── getting-started.md
├── tutorials/quickstart.md
├── api/
└── architecture/
examples/**/*.ipynb
tests/test_notebooks.py
.github/workflows/{docs.yml,ci.yml}
README.md
CITATION.cff
.zenodo.json
RELEASING.md
```

**Structure Decision**: Single-project library. Notebooks remain under `examples/`; MkDocs copies them into `docs/examples/` at build time. Generated copy is gitignored.

## Phase 0 research

- Hatchling src-layout auto-discovery: omitting `packages = ["src/xeries"]` avoids a wheel whose import path is `src.xeries`.
- `mkdocs-jupyter` only reads files under `docs_dir`; a pre-build copy hook is required.
- Executing notebooks during every docs CI run is slow and path-noisy; commit outputs and set `execute: false`.
- PEP 735 groups vs extras: keep published extras in `[project.optional-dependencies]`; local-only `dev` and `docs` groups; install notebooks via `--extra notebooks`.

## Complexity Tracking

> No constitution violations.

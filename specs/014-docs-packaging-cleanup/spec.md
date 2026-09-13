# Feature Specification: Docs and packaging cleanup

**Feature Branch**: `014-docs-packaging-cleanup`  
**Created**: 2026-09-14  
**Status**: Draft  
**Home repo**: xeries  
**Input**: User description: cleanup unnecessary files, fix wrong documentation links, include example notebooks in the docs site with rendered results, and fix dependency management in `pyproject.toml`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Working documentation site (Priority: P1)

A reader opens the published MkDocs site (or a local `mkdocs serve`) and can follow every nav item and in-page link to a real page. Architecture, tutorials, API reference, and example notebooks all resolve. Example notebooks show saved tables and plots, not empty cells.

**Why this priority**: Broken nav and missing notebooks make the library unusable from the docs site.

**Independent Test**: `uv run mkdocs build --strict` succeeds; every nav path exists; example notebook pages contain executed outputs.

**Acceptance Scenarios**:

1. **Given** the MkDocs nav, **When** the site is built with `--strict`, **Then** no missing-file warnings fail the build.
2. **Given** a notebook listed under Examples, **When** the page is opened, **Then** it is the corresponding file from `examples/` rendered with committed cell outputs.
3. **Given** getting-started and quickstart tutorial code samples, **When** a reader copies them, **Then** they call `explain()`, not the non-existent `compute()`.

---

### User Story 2 - Honest shipped vs planned surface (Priority: P2)

A reader can tell which explainers are available now (`ConditionalPermutationImportance`, `ConditionalSHAP`, hierarchy, visualization) versus planned (SHAP-IQ, feature dropping, causal importance, Darts adapter).

**Why this priority**: Docs currently mark shipped SHAP/hierarchy as planned, which contradicts `specs/005` and `specs/006`.

**Independent Test**: Home, getting-started, README method table, and API overview agree with the public `__all__` in `src/xeries/__init__.py`.

**Acceptance Scenarios**:

1. **Given** the docs home and README, **When** a reader looks up Conditional SHAP, **Then** it is listed as available.
2. **Given** the API reference, **When** mkdocstrings renders explainer members, **Then** `explain` is documented and `compute` is not listed as a member.

---

### User Story 3 - Installable package and extras (Priority: P1)

A contributor clones the repo, runs `uv sync`, and gets a consistent environment. A user installing from PyPI can use documented extras. The wheel contains the `xeries` package at the import root (not `src/xeries`). Project metadata URLs point at `xeries-labs`.

**Why this priority**: Hatch wheel layout, duplicated extras/groups, `lightgbm` in default `dev`, ignored lockfile, and stale GitHub URLs make installs and CI non-reproducible.

**Independent Test**: `uv lock` succeeds; `uv build` wheel listing shows `xeries/` modules; `pip show` / `project.urls` use `xeries-labs`; `lightgbm` is only in the `notebooks` extra.

**Acceptance Scenarios**:

1. **Given** `pyproject.toml`, **When** a wheel is built, **Then** importable files live under `xeries/`, not `src/xeries/`.
2. **Given** default `uv sync`, **When** the environment is created, **Then** `lightgbm` is not a required runtime or default-dev package.
3. **Given** project metadata, **When** a user follows Homepage/Documentation/Repository, **Then** they land on `xeries-labs/xeries` and `xeries-labs.github.io/xeries`.

---

### User Story 4 - Notebook tests match the tree (Priority: P2)

CI notebook tests discover the nested notebooks under `examples/` instead of looking for obsolete root-level names.

**Why this priority**: Current tests skip silently, so broken notebooks would not fail CI.

**Independent Test**: `pytest tests/test_notebooks.py -m "notebook and not slow"` finds every `*.ipynb` under `examples/` (excluding checkpoints).

**Acceptance Scenarios**:

1. **Given** notebooks in `examples/quickstart/` and siblings, **When** structure tests run, **Then** they validate those files rather than skipping because `examples/*.ipynb` is empty.

---

### Edge Cases

- MkDocs must not require files outside `docs/` at configure time; notebooks are copied in `on_pre_build`.
- Generated `docs/examples/` must not be committed.
- Historical research text in governance (`roadmap-initial.md`) is not edited.
- No public symbols are added or removed under `src/xeries/`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: MkDocs nav MUST only reference files that exist after the pre-build hook (markdown under `docs/` plus copied notebooks).
- **FR-002**: Example notebooks MUST remain the source of truth under `examples/` and MUST appear on the docs site with saved outputs.
- **FR-003**: User-facing docs MUST use `explain()` for permutation importance.
- **FR-004**: Docs MUST describe Conditional SHAP and hierarchical explainers as shipped; SHAP-IQ, dropping, causal, and Darts as planned.
- **FR-005**: `project.urls`, README, CITATION.cff, `.zenodo.json`, and RELEASING MUST use the `xeries-labs` GitHub org and Pages URL.
- **FR-006**: Hatchling MUST publish a src-layout wheel without embedding a top-level `src/` directory.
- **FR-007**: `lightgbm` MUST NOT be a core or default `dev` dependency; it MAY appear in the `notebooks` extra.
- **FR-008**: Notebook optional-dependencies MUST NOT be duplicated as a second identical dependency group; `uv sync --extra notebooks` is the install path.
- **FR-009**: `uv.lock` MUST be tracked (not gitignored).
- **FR-010**: `docs/Initial_plan.md` MUST be removed from this repo (duplicate of governance `roadmap-initial.md`).
- **FR-011**: Notebook tests MUST recurse into `examples/` subdirectories.
- **FR-012**: Docs CI MUST run `mkdocs build --strict`.
- **FR-013**: This spec MUST NOT change the public Python API under `src/xeries/`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `uv run mkdocs build --strict` exits 0.
- **SC-002**: Every Examples nav entry renders a notebook that contains at least one non-empty code-cell output.
- **SC-003**: Grep of `docs/` (excluding deleted Initial_plan) finds no `explainer.compute` / `members: compute` for permutation importance.
- **SC-004**: Built wheel listing contains `xeries/__init__.py` and does not contain `src/xeries/__init__.py`.
- **SC-005**: `pytest tests/test_notebooks.py -m "notebook and not slow"` reports collected notebooks equal to the number of `examples/**/*.ipynb` files.

## Assumptions

- Governance submodule `.specify/` is not edited from this repo; adding a roadmap row is a follow-up PR on `xeries-labs/xeries-governance`.
- Notebooks are re-executed locally and committed with outputs; docs build does not execute notebooks (`mkdocs-jupyter` `execute: false`).
- Specs 001–007 public contracts stay unchanged.
- Partial edits already present on the working tree (mkdocs nav, hooks, some API pages, pyproject extras) are completed and verified as part of this spec, not treated as a separate project.

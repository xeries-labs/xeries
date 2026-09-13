# Tasks: Docs and packaging cleanup

**Input**: Design documents from `/specs/014-docs-packaging-cleanup/`  
**Prerequisites**: [spec.md](./spec.md) (required), [plan.md](./plan.md) (required)

**Tests**: Notebook discovery tests and `mkdocs build --strict`. No new public-symbol contract tests (FR-013).

**Organization**: Tasks grouped by user story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1–US4 from spec.md

## Path Conventions

All paths relative to the `xeries` repo root. Do not edit `.specify/`.

---

## Phase 1: Setup

- [x] T001 Create `specs/014-docs-packaging-cleanup/{spec,plan,tasks}.md` and list the spec in `specs/README.md` active queue

---

## Phase 2: Foundational (packaging)

- [x] T002 [US3] Fix `pyproject.toml`: `project.urls` → xeries-labs; drop hatch `packages`; extras vs groups; `lightgbm` only in `notebooks` extra; `[tool.uv] default-groups = ["dev"]`
- [x] T003 [US3] Stop ignoring `uv.lock` in `.gitignore`; gitignore generated `docs/examples/`; run `uv lock`
- [x] T004 [P] [US3] Update install snippets in `README.md` and `examples/README.md` (`uv sync --extra notebooks`, `--group docs`)

---

## Phase 3: User Story 1 — Working documentation site

- [x] T005 [US1] Point `mkdocs.yml` nav at real architecture/API files and copied notebook paths; `mkdocs_hooks.py` copies from `examples/`
- [x] T006 [P] [US1] Replace `.compute(` with `.explain(` in `docs/getting-started.md` and `docs/tutorials/quickstart.md`
- [x] T007 [US1] Add `docs/examples.md`; copy notebooks via hook; `mkdocs-jupyter` `execute: false`
- [x] T008 [US1] Re-execute `examples/**/*.ipynb`, fix leftover `tcpfi` strings, strip machine-specific warning paths from outputs

---

## Phase 4: User Story 2 — Honest shipped vs planned surface

- [x] T009 [P] [US2] Update `docs/index.md`, `docs/api/reference.md`, `docs/api/importance.md`, README method table: SHAP/hierarchy shipped; SHAP-IQ/dropping/causal/Darts planned; mkdocstrings members `explain` not `compute`
- [x] T010 [P] [US2] Add API pages `docs/api/hierarchy.md` and `docs/api/visualization.md` if missing
- [x] T011 [P] [US3] Fix `thec0dewriter` URLs in `CITATION.cff`, `.zenodo.json`, `RELEASING.md`, `README.md`

---

## Phase 5: User Story 4 — Notebook tests and CI

- [x] T012 [US4] Fix `tests/test_notebooks.py` to `rglob("*.ipynb")` and parametrize real nested filenames
- [x] T013 [US1] Docs workflow: `uv sync --group docs` then `mkdocs build --strict`; notebook CI uses `--extra notebooks`

---

## Phase 6: Hygiene

- [x] T014 [US1] Delete `docs/Initial_plan.md`
- [x] T015 Verify: `uv build` wheel layout; notebook structure tests; `uv run mkdocs build --strict`

---

## Dependencies

- T001 before everything
- T002–T003 before T015
- T005–T008 before T015
- T014 before T015 (strict build must not still list Initial_plan)

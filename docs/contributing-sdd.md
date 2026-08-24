# Contributing under Spec-Driven Development

`xeries` is part of the **[xeries-labs](https://github.com/xeries-labs)** Spec-Driven Development (SDD) program. The constitution, the program-wide roadmap, and the spec / plan / tasks templates all live in a single governance repository, [`xeries-labs/xeries-governance`](https://github.com/xeries-labs/xeries-governance), and are consumed here as a git submodule mounted at `.specify/`.

This page is the contributor-facing companion to [`AGENTS.md`](https://github.com/xeries-labs/xeries/blob/main/AGENTS.md). Read it before opening a non-trivial PR.

## The five binding principles

The full text lives in `.specify/memory/constitution.md`. The five program-wide principles you will be asked to honour at every gate:

1. **I. Specs before code.** Every change under `src/`, `tests/`, or `docs/` traces to a spec in `specs/NNN-<slug>/`. Backfilled specs (`specs/001-…` to `specs/007-…`) cover the existing surface; new features start with a fresh spec.
2. **II. Agent-agnosticism.** No Cursor-only or Copilot-only constructs in the runtime tree. Per-agent shims live under per-agent directories (`.github/`, `.cursor/`).
3. **III. Test-first for new contracts (NON-NEGOTIABLE).** A failing contract test ships **before** the implementation commit for any new public symbol.
4. **IV. Typed public surface.** `ty check src` and `ruff check src tests` MUST pass. Public symbols carry full type annotations.
5. **V. Reproducibility discipline (bench only).** Applies to `xeries-bench`, not to this repo. We mention it here only because it shapes the Repo-scope rules below.

## Repo scope — what belongs where

| Concern | Lives in |
| --- | --- |
| Public XAI library (partitioners, importance, adapters, hierarchy, visualisation) | **`xeries`** (this repo) |
| Synthetic-interaction harness, global LightGBM training, TreeSHAP-IQ benchmarks, sign-agreement / FP audits | [`xeries-labs/xeries-bench`](https://github.com/xeries-labs/xeries-bench) |
| Constitution, roadmap, spec / plan / tasks templates, agent shims | [`xeries-labs/xeries-governance`](https://github.com/xeries-labs/xeries-governance) (submoduled at `.specify/`) |

Practical consequences for PRs against this repo:

- **Do not** add `lightgbm`, `catboost`, or any heavy GBM as a runtime dep. They live in `xeries-bench`. (`xeries[shapiq]` is fine; `shapiq` itself stays an optional extra here.)
- **Do not** commit large data, parquet/feather/npz/h5, or rendered notebooks with embedded multi-MB figures.
- **Do not** edit anything under `.specify/`. That folder is governed by `xeries-labs/xeries-governance` and edits MUST go through a PR there.

## The contributor workflow

```mermaid
flowchart LR
  A[Idea] --> B{Existing spec covers it?}
  B -- yes --> C[Cite Backfilled spec\nin commit message]
  B -- no --> D[Author specs/NNN-…/spec.md]
  D --> E[Author specs/NNN-…/plan.md\nrun Constitution Check]
  E --> F{Gate passes?}
  F -- no --> G[Add Complexity Tracking\nentry OR redesign]
  G --> E
  F -- yes --> H[specs/NNN-…/tasks.md]
  H --> I[Implement test-first\n(Principle III)]
  C --> I
  I --> J[uv run ruff/ty/pytest/mkdocs]
  J --> K[Open PR]
```

### 1. Find or author a spec

For mechanical refactors that preserve the public surface, you do not need a new spec — cite the affected Backfilled spec(s) in your commit message and PR description. The Backfilled queue:

| Spec ID | Module |
| --- | --- |
| `001-core-contracts` | `xeries.core` |
| `002-partitioners` | `xeries.partitioners` |
| `003-conditional-permutation-importance` | `xeries.importance.permutation` |
| `004-framework-adapters` | `xeries.adapters` |
| `005-conditional-shap` | `xeries.importance.shap` |
| `006-hierarchical-explainer` | `xeries.hierarchy` |
| `007-visualization` | `xeries.visualization` |

For anything that adds, removes, or breaks a public symbol, author a new spec:

```bash
mkdir -p specs/NNN-short-slug
cp .specify/templates/spec-template.md specs/NNN-short-slug/spec.md
# Fill in the Home repo (xeries) and the User Stories sections.
# Add a row in `.specify/memory/roadmap.md` via a PR to the governance repo.
```

### 2. Plan and run the Constitution Check

```bash
cp .specify/templates/plan-template.md specs/NNN-short-slug/plan.md
```

The plan template carries a **Constitution Check** table. Every "Needs justification" verdict requires a Complexity Tracking entry with a Simpler Alternative. Without that, the gate fails and Phase 0 cannot start.

### 3. Break into tasks, then implement test-first

```bash
cp .specify/templates/tasks-template.md specs/NNN-short-slug/tasks.md
```

Implement strictly test-first when you introduce new public symbols. Mechanical refactors are exempt from Principle III, but still need green tests.

### 4. Local verification before pushing

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run ty check src
uv run pytest
uv run mkdocs build --strict
```

Each of these is also gated in CI.

### 5. Open the PR

The PR description should at minimum link to the spec(s), summarise the change, and call out any "Needs justification" entries from the Constitution Check.

## Slash commands (Copilot)

If you are running through GitHub Copilot, the `.github/prompts/` shims map the Spec Kit 0.8 slash commands onto this repo. The most useful ones during a contribution:

- `/speckit.specify` — author a new `spec.md` from the template.
- `/speckit.plan` — produce `plan.md` and run the Constitution Check.
- `/speckit.tasks` — break the plan into actionable tasks.
- `/speckit.implement` — execute the task list.
- `/speckit.checklist` — quality checklist for a spec or plan.
- `/speckit.analyze` — cross-artefact consistency audit.

Other agents (Cursor, Claude, etc.) read the same workflows from `.github/agents/*.agent.md`.

## Upgrading the governance submodule

When `xeries-labs/xeries-governance` cuts a new tag (e.g. `v1.1.0`):

```bash
git -C .specify fetch --tags
git -C .specify checkout v1.1.0
git add .specify
git commit -m "chore(sdd): bump xeries-governance to v1.1.0"
```

This is the **only** way the rules in `.specify/` change in this repo. Editing files under `.specify/` directly is not a supported workflow.

## Where to ask questions

- A repo-scope question: `.specify/memory/constitution.md`, § Repo scope.
- A workflow question: this page.
- A roadmap question (is feature X planned?): `.specify/memory/roadmap.md`.
- Anything else: open an issue at [`xeries-labs/xeries`](https://github.com/xeries-labs/xeries/issues).

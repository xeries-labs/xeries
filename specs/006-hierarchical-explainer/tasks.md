# Tasks: Hierarchical explainer (`xeries.hierarchy`)

**Input**: Design documents from `/specs/006-hierarchical-explainer/`
**Prerequisites**: [spec.md](./spec.md) (required), [plan.md](./plan.md) (required)

**Tests**: Tests are present and required — they were authored alongside the
implementation in commit `2a5d7f6`. Every task that maps to a public symbol
cites its matching contract test class as evidence.

**Organization**: Tasks are grouped by user story (matching `spec.md`) so
each story is independently verifiable. **All tasks except T014 are
`[Backfilled]` — they are evidence-trace pointers into the existing
implementation, not actionable work for this PR.** T014 is the only task
this PR actually executes.

> **Status: Backfilled from implementation** — the `xeries.hierarchy`
> subpackage was implemented before SDD adoption. The tasks below describe
> the implementation as it exists on the branch tip at the time of writing,
> using the standard SDD task shape so the spec / plan / tasks triplet is
> coherent for future contributors.

## Format: `[ID] [P?] [Story] [Backfilled?] Description`

- **[P]**: Can run in parallel (different files, no dependencies). Only
  meaningful for the single non-backfilled task this round.
- **[Story]**: Which user story this task belongs to (US1, US2). Foundational
  and setup tasks have no story label.
- **[Backfilled]**: Task corresponds to already-shipped code; the citation
  is the evidence trace.

<!--
  xeries-labs reminder:
  - Every task MUST stay inside the Home repo declared in the parent spec.md / plan.md.
  - Tasks that touch files under `.specify/` are a red flag: that folder is governed by the
    `xeries-governance` submodule and MUST NOT be edited from consumer repos. Route any
    such change to a PR against `xeries-labs/xeries-governance` instead.
-->

## Path Conventions

- **Single project** (per `plan.md` Structure Decision): `src/xeries/`,
  `tests/` at repository root.
- All paths below are relative to the `xeries` repo root.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project-level scaffolding that the `xeries.hierarchy` package
depends on. All items already on `main`.

- [x] T001 [Backfilled] Confirm runtime deps `numpy`, `pandas`,
  `scikit-learn`, `shap>=0.49.1` listed in `pyproject.toml`
  (lines 35-40). No new deps required for hierarchy — uses only the
  already-mandated minimum floor from the constitution's tech-stack table.
- [x] T002 [P] [Backfilled] Confirm tool configs in place: `ty.toml`
  (type-checker config, governs `ty check src`), `pyproject.toml`
  `[tool.ruff]` block, `.pre-commit-config.yaml` (25 LoC). All inherited
  from the repo's existing setup; no hierarchy-specific entries needed.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data classes and the hierarchy abstraction itself. Must
exist before either user story is operational.

⚠️ **CRITICAL**: All foundational tasks are Backfilled — implementation is
on `main` at commit `2a5d7f6` (or its descendant on `feat/sdd-adoption`).

- [x] T003 [Backfilled] `src/xeries/hierarchy/__init__.py` — package
  docstring + four re-exports (`HierarchicalAggregator`,
  `HierarchicalExplainer`, `HierarchicalResult`, `HierarchyDefinition`).
  Evidence: 40 LoC; `__all__` list lines 36-41.
- [x] T004 [P] [Backfilled] `src/xeries/hierarchy/types.py` —
  `HierarchicalResult` dataclass with fields `levels`, `features`,
  `importance_by_level`, `raw_values_by_level`, `feature_values_by_level`,
  `method`, and accessors `get_level_df`, `get_global`,
  `get_cohorts_at_level`, `get_raw_values`. Evidence: 148 LoC; contract
  tests `tests/hierarchy/test_types.py::TestHierarchicalResult` (176 LoC
  total, 2 test classes).
- [x] T005 [P] [Backfilled] `src/xeries/hierarchy/definition.py` —
  `HierarchyDefinition` class supporting three cohort-resolution
  strategies: column-based (`columns=[...]`), regex `parse_pattern`, and
  explicit `mapping`; plus `get_cohorts(X, level)`, `validate_data(X)`,
  `add_hierarchy_columns(X)`. Evidence: 307 LoC; contract tests
  `tests/hierarchy/test_definition.py` (265 LoC; 5 test classes
  `TestHierarchyDefinition{Init, ColumnBased, ParseBased, ExplicitMapping,
  Repr}`).
- [x] T006 [Backfilled] Top-level re-exports in `src/xeries/__init__.py`
  (lines 60-64 import, 94-98 `__all__`) — `HierarchicalAggregator`,
  `HierarchicalExplainer`, `HierarchicalResult`, `HierarchyDefinition`
  are reachable as `from xeries import HierarchicalExplainer, ...`.

**Checkpoint**: Foundation is on `main`; user stories below are testable
independently against it.

---

## Phase 3: User Story 1 — Aggregate SHAP up an M5-style hierarchy (Priority: P1) 🎯 MVP

**Goal**: Given a `HierarchyDefinition`, a fitted `ConditionalSHAP` base
explainer, and an `X_test` DataFrame, produce per-level feature-importance
tables (one row per `(cohort, feature)` at each level, plus a `"global"`
roll-up) via `HierarchicalExplainer.explain(X_test)`.

**Independent Test** (from `spec.md`, US1):

```python
hierarchy = HierarchyDefinition(
    levels=["state", "store"],
    columns=["state_id", "store_id"],
)
base = ConditionalSHAP(model, X_train, series_col="level")
explainer = HierarchicalExplainer(base, hierarchy)
result = explainer.explain(X_test)
result.get_global()           # one row per feature
result.get_level_df("state")  # one row per (state, feature)
```

### Contract tests for User Story 1

- [x] T007 [P] [US1] [Backfilled] `tests/hierarchy/test_aggregator.py::TestHierarchicalAggregatorSHAP`
  — asserts that mean-absolute-SHAP per cohort matches the closed-form
  formula in plan.md §Aggregation rules. Evidence: lines 15-176 of
  `tests/hierarchy/test_aggregator.py`.
- [x] T008 [P] [US1] [Backfilled] `tests/hierarchy/test_explainer.py::TestHierarchicalExplainerInit`
  and `TestHierarchicalExplainerExplain` — assert `explain(X)` returns a
  `HierarchicalResult` with `result.levels == [*hierarchy.levels,
  "global"]` and every level's DataFrame indexed by cohort. Evidence:
  lines 46-156 of `tests/hierarchy/test_explainer.py`, MockSHAPExplainer
  defined lines 18-44.

### Implementation for User Story 1

- [x] T009 [US1] [Backfilled] `src/xeries/hierarchy/aggregator.py::HierarchicalAggregator.aggregate_shap`
  — implements the SHAP roll-up path. Evidence: `aggregator.py` 254 LoC
  total; method matches FR-006.2 of `spec.md`.
- [x] T010 [US1] [Backfilled] `src/xeries/hierarchy/explainer.py::HierarchicalExplainer.explain`
  — orchestrator: dispatches to the base explainer, hands the raw
  attribution result to `HierarchicalAggregator`, returns
  `HierarchicalResult`. Includes `include_raw=True` flag per FR-006.3.
  Evidence: `explainer.py` 281 LoC; result-type tests in
  `tests/hierarchy/test_types.py::TestHierarchicalResult` (176 LoC).

**Checkpoint**: US1 is fully operational on `main`. Acceptance scenarios
1 and 2 of `spec.md` US1 are covered by T007 and T008.

---

## Phase 4: User Story 2 — Compare permutation and SHAP at the store level (Priority: P2)

**Goal**: Given two `HierarchicalExplainer` instances built on different
base explainers (one `ConditionalSHAP`, one `ConditionalPermutationImportance`),
produce per-cohort importance tables at the `store` level on each, with
matching `(store, feature)` indices so the two rankings are directly
comparable.

**Independent Test** (from `spec.md`, US2): build two `HierarchicalExplainer`s
with different base explainers, call `result.get_level_df("store")` on
both, assert the two DataFrames have the same `(store, feature)` index.

### Contract tests for User Story 2

- [x] T011 [P] [US2] [Backfilled] `tests/hierarchy/test_aggregator.py::TestHierarchicalAggregatorImportance`
  — asserts that `aggregate_importance(result, X, y)` re-runs the base
  explainer per cohort and returns a `HierarchicalResult` whose
  `get_level_df(level)` has the same cohort index as the SHAP path.
  Evidence: lines 177-229 of `tests/hierarchy/test_aggregator.py`.
- [x] T012 [P] [US2] [Backfilled] `tests/hierarchy/test_explainer.py::TestHierarchicalExplainerCompare`
  and `TestHierarchicalExplainerExplainLevel` — assert
  `compare_cohorts(X, level="store")` and `explain_level(X, level, cohort)`
  produce comparable, ordered outputs. Evidence: lines 158-298 of
  `tests/hierarchy/test_explainer.py`.

### Implementation for User Story 2

- [x] T013 [US2] [Backfilled] `src/xeries/hierarchy/aggregator.py::HierarchicalAggregator.aggregate_importance`
  + `src/xeries/hierarchy/explainer.py::HierarchicalExplainer.compare_cohorts`
  / `explain_level` / `explain_per_cohort` / `feature_ranking_stability`.
  Evidence: methods named in `docs/architecture/hierarchy.md` lines 29-37;
  tested by T011 and T012.

**Checkpoint**: US2 is operational on `main`. The two user stories work
independently and can be combined freely.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, traceability, and the one piece of net-new work
this PR introduces.

- [ ] T014 Cross-link the spec / plan / tasks triplet: append a "See also"
  line to `specs/006-hierarchical-explainer/spec.md` pointing at
  [plan.md](./plan.md) and [tasks.md](./tasks.md). **This is the only
  non-backfilled task this PR adds.** No code changes.
- [x] T015 [P] [Backfilled] Architecture reference at
  `docs/architecture/hierarchy.md` (312 LoC) — class diagram, hierarchy
  strategies, aggregation formula, sequence diagram, full example
  workflow. Already on the branch.
- [x] T016 [P] [Backfilled] Roadmap row in
  `.specify/memory/roadmap.md` (via the `xeries-governance` submodule
  pinned to v1.0.0) marks
  `006-hierarchical-explainer` as `Backfilled`, Home repo `xeries`. Note
  per Principle II / constitution Repo-scope table: the governance
  submodule is **not** edited from this repo; the row already exists on
  the pinned tag, so this task is verification-only.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Already shipped on `main`. No-op for this PR.
- **Foundational (Phase 2)**: Already shipped on `main`. No-op for this PR.
- **User Story 1 (P1)**: Already shipped on `main`. No-op for this PR.
- **User Story 2 (P2)**: Already shipped on `main`. No-op for this PR.
- **Polish (Phase 5)**: T014 is the only active task this PR runs.

### Within-PR execution order

For this PR specifically (markdown-only):

1. Author `plan.md` (done in Phase 5 sibling task `author_plan_md`).
2. Author this `tasks.md` (current task).
3. Execute T014 — append "See also" line to `spec.md`.
4. Run quality gates locally (`ruff check`, `ty check`, `pytest`).
5. Push branch, open PR.

### Parallel Opportunities

- T004 and T005 historically landed in the same commit but as separate
  files; they would be `[P]` if this were a fresh implementation.
- T007 and T008 are independent test files / classes.
- T011 and T012 are independent test classes within the same file family.
- T015 and T016 are independent docs / submodule pointers.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

The MVP path defined in `spec.md` US1 is already shipped: a user can
construct a `HierarchyDefinition`, wrap a `ConditionalSHAP` base in
`HierarchicalExplainer`, call `.explain(X_test)`, and read
`result.get_global()` plus `result.get_level_df("state")`.

### Incremental Delivery

US1 (SHAP roll-up) and US2 (permutation-vs-SHAP comparison) are both
shipped. No further increments are planned in this PR; future hierarchy
work (e.g. `HierarchicalCausalExplainer`) requires a fresh spec.

---

## Notes

- [P] tasks = different files, no dependencies.
- [Story] label maps task to specific user story for traceability.
- [Backfilled] label means: the task is evidence-trace into already-shipped
  code, not actionable work for this PR.
- Each user story (US1, US2) is independently completable and testable —
  US2 does not depend on US1's code path, only on the shared foundational
  classes from Phase 2.
- The only new artefacts in this PR are `plan.md`, `tasks.md`, and the
  one-line "See also" addition to `spec.md` (task T014). Everything else
  is verification of prior work.

# Feature Specification: Hierarchical explainer (`xeries.hierarchy`)

**Feature Branch**: `feat/sdd-adoption` (backfill)  
**Created**: 2026-04-26  
**Status**: Backfilled from implementation  
**Home repo**: xeries  
**Input**: SDD bootstrap — describe the existing public surface of `xeries.hierarchy`.

> **Status: Backfilled from implementation** — this spec describes the code as it
> exists on `main` at commit `858a498`. Redesigns require a follow-up spec.

## Summary

Real-world demand-forecasting datasets (M5, retail, supply-chain) come with a
**hierarchy of series**: country → state → store → product. A single global
model fits all leaves at once, and stakeholders want feature-importance
reports at every level of the hierarchy — not just per-leaf or fully global.

The `xeries.hierarchy` subpackage takes any base explainer (`ConditionalSHAP`,
`ConditionalPermutationImportance`, …) and lifts it into a hierarchical
explainer that aggregates importance up the tree.

## User Scenarios & Testing

### User Story 1 — Aggregate SHAP up an M5-style hierarchy (Priority: P1)

A user wants the same set of features ranked at country / state / store
levels.

**Independent Test**:

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

**Acceptance Scenarios**:

1. **Given** a `HierarchyDefinition` with `levels=["state", "store"]`,
   **When** `explain` is called, **Then** `result.levels` returns
   `["state", "store"]` plus an implicit `"global"` aggregation.
2. **Given** the base explainer returns a `SHAPResult` with `series_ids`
   filled in, **When** `HierarchicalAggregator` aggregates, **Then** each
   level's per-cohort mean-absolute SHAP equals the average of the
   corresponding leaves.

### User Story 2 — Compare permutation and SHAP at the store level (Priority: P2)

A user wants to plot, side by side, cPFI and SHAP rankings at the store
level.

**Independent Test**: build two `HierarchicalExplainer`s with different base
explainers, call `result.get_level_df("store")` on both, assert the two
DataFrames have the same `(store, feature)` index.

## Functional Requirements

### FR-006.1 `HierarchyDefinition`

```python
HierarchyDefinition(
    levels: list[str],
    columns: list[str],
)
```

- `levels` is the ordered list of aggregation levels (highest to lowest).
- `columns` maps each level to the column / index level in `X` that
  identifies cohorts at that level.
- A `"global"` level is implicit and always available.
- Validates that `len(levels) == len(columns)` and that the lists are
  non-empty.

### FR-006.2 `HierarchicalAggregator`

- Takes a per-row attribution / importance vector and a `HierarchyDefinition`,
  returns a dict mapping each level to a `pandas.DataFrame` indexed by cohort
  (with `feature` columns).
- Aggregation rule for SHAP-like attributions: per-cohort mean of absolute
  SHAP values.
- Aggregation rule for permutation-style importances: importance is recomputed
  per cohort by re-running the base explainer on each cohort's rows.

### FR-006.3 `HierarchicalExplainer`

```python
HierarchicalExplainer(
    base_explainer: BaseExplainer,
    hierarchy: HierarchyDefinition,
)
```

- `explain(X, *, include_raw: bool = False, ...)`: returns a
  `HierarchicalResult`. If `include_raw=True`, the underlying base result
  (e.g. `SHAPResult`) is also embedded.
- The hierarchical explainer MUST work with both `AttributionExplainer`
  subclasses (per-row attributions) and `MetricBasedExplainer` subclasses
  (per-cohort metric drops).

### FR-006.4 `HierarchicalResult`

- `levels: list[str]` — including `"global"`.
- `get_global() -> pd.DataFrame` — feature ranking aggregated across all
  cohorts.
- `get_level_df(level: str) -> pd.DataFrame` — `(cohort, feature)` indexed
  ranking at the given level.
- `to_dataframe() -> pd.DataFrame` — long-format with `level` /
  `cohort` / `feature` / `importance` columns.

## Public API surface

```python
from xeries.hierarchy import (
    HierarchyDefinition,
    HierarchicalAggregator,
    HierarchicalExplainer,
    HierarchicalResult,
)
# Re-exported at top level:
from xeries import (
    HierarchyDefinition,
    HierarchicalAggregator,
    HierarchicalExplainer,
    HierarchicalResult,
)
```

Source files:

- `src/xeries/hierarchy/definition.py`
- `src/xeries/hierarchy/aggregator.py`
- `src/xeries/hierarchy/explainer.py`
- `src/xeries/hierarchy/types.py`

## Test coverage

- `tests/hierarchy/test_definition.py` — validation of the hierarchy schema.
- `tests/hierarchy/test_aggregator.py` — per-level aggregation correctness.
- `tests/hierarchy/test_explainer.py` — end-to-end with a mock base
  explainer.
- `tests/hierarchy/test_types.py` — `HierarchicalResult` accessors.

## Out of scope / future work

- Reconciliation of forecasts across the hierarchy (the
  `MinT`/`OLS` family) is out of scope — `xeries` focuses on importance,
  not on point forecasts.
- A `HierarchicalCausalExplainer` is on the long-term roadmap but blocked
  by the absence of a `CausalFeatureImportance` base explainer.

## See also

- [plan.md](./plan.md) — implementation plan + Constitution Check (Backfilled).
- [tasks.md](./tasks.md) — task-level evidence trace into the existing
  `xeries.hierarchy` implementation.

# Feature Specification: Partitioners (`xeries.partitioners`)

**Feature Branch**: `feat/sdd-adoption` (backfill)  
**Created**: 2026-04-26  
**Status**: Backfilled from implementation  
**Home repo**: xeries  
**Input**: SDD bootstrap — describe the existing public surface of `xeries.partitioners`.

> **Status: Backfilled from implementation** — this spec describes the code as it
> exists on `main` at commit `858a498`. Redesigns require a follow-up spec.

## Summary

A **partitioner** turns the input feature DataFrame into a vector of integer
group labels, one per sample. These labels feed conditional permutation
importance: a feature's values are shuffled **only within a group**, preserving
the conditional distribution of correlated features. Two strategies ship:

- `ManualPartitioner` — user supplies a dict mapping series-IDs (or any
  categorical) to group labels.
- `TreePartitioner` — a `DecisionTreeRegressor` is fitted to predict the
  feature of interest from all other features; the tree's leaf indices become
  the group labels (the cs-PFI algorithm of Molnar et al.).

## User Scenarios & Testing

### User Story 1 — Domain expert encodes known cohorts (Priority: P1)

A demand-forecasting analyst knows that products MT_001 and MT_003 share
demand patterns while MT_002 is distinct.

**Independent Test**: instantiate
`ManualPartitioner({"MT_001": "A", "MT_002": "B", "MT_003": "A"})`, fit on a
DataFrame indexed by series, assert `n_groups == 2` and that samples with
matching series-IDs receive identical integer labels.

**Acceptance Scenarios**:

1. **Given** a mapping covering all series in `X`, **When** `fit_get_groups`
   is called, **Then** every sample receives a label from `range(n_groups)`.
2. **Given** a mapping that is missing a series-ID present in `X`, **When**
   `get_groups` is called, **Then** a `ValueError` is raised naming the
   missing IDs.
3. **Given** the partitioner has not been fitted, **When** `get_groups` is
   called, **Then** a `ValueError` is raised.

### User Story 2 — Automated subgroup discovery (cs-PFI) (Priority: P1)

A user has many correlated lags and no obvious cohort structure. They want
the library to discover homogeneous subgroups automatically.

**Independent Test**: fit `TreePartitioner(max_depth=4)` on a DataFrame whose
target feature is highly predictable from other columns; assert that
`n_groups > 1` and that within each group the variance of the target feature
is materially smaller than the global variance.

**Acceptance Scenarios**:

1. **Given** a DataFrame with `_level_skforecast` or `level` column, **When**
   `series_col=None`, **Then** the partitioner auto-detects the column and
   one-hot-encodes it before fitting the tree.
2. **Given** `min_samples_leaf=0.05` (default), **When** the tree fits on a
   small DataFrame, **Then** every leaf contains at least 5% of samples.
3. **Given** `random_state=0`, **When** `fit_get_groups` is called twice on
   the same input, **Then** the two label arrays are identical.

## Functional Requirements

### FR-002.1 `ManualPartitioner`

- `__init__(mapping: dict, series_col: str = "level")`.
- `fit(X, feature)` builds an internal label encoder from
  `sorted(set(mapping.values()))` and sets `_fitted = True`.
- `get_groups(X)` extracts series IDs from either `X.index.get_level_values(series_col)`
  (for `MultiIndex`) or `X[series_col]`, applies `mapping`, then encodes to `np.intp`.
- Raises `KeyError` if `series_col` is not found in either index or columns.
- `n_groups` property returns the number of unique encoded labels.

### FR-002.2 `TreePartitioner`

- `__init__(max_depth=4, min_samples_leaf=0.05, series_col=None,
  random_state=None)`.
- `fit(X, feature)` trains
  `DecisionTreeRegressor(max_depth, min_samples_leaf, random_state)` on
  `X.drop(columns=[feature])` (with `series_col` one-hot-encoded if present)
  predicting `X[feature]`.
- `get_groups(X)` returns `tree.apply(X_tree)` cast to `np.intp`.
- Auto-detection rule for `series_col=None`: try
  `_level_skforecast` first (skforecast 0.21+), then `level` (legacy /
  MultiIndex).

### FR-002.3 Both partitioners

- Implement `BasePartitioner` (FR-002, FR-003 in spec 001).
- Return `NDArray[np.intp]` from `get_groups`.
- Are deterministic for a fixed `random_state`.

## Public API surface

```python
from xeries.partitioners import ManualPartitioner, TreePartitioner
# Re-exported at top level:
from xeries import ManualPartitioner, TreePartitioner
```

Source files: `src/xeries/partitioners/manual.py`,
`src/xeries/partitioners/tree.py`.

## Test coverage

- `tests/unit/test_partitioners.py` covers both partitioners' fit/predict
  cycles, fitted-state errors, and the cs-PFI determinism guarantee.

## Out of scope / future work

- A `KMeansPartitioner` and a `QuantileBinPartitioner` were considered but
  are not on the active roadmap.
- The auto-detection logic for `series_col` is currently informal; if a third
  column convention emerges, a follow-up spec MUST formalise the resolution
  order.

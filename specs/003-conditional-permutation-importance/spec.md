# Feature Specification: Conditional Permutation Importance (`xeries.importance.permutation`)

**Feature Branch**: `feat/sdd-adoption` (backfill)  
**Created**: 2026-04-26  
**Status**: Backfilled from implementation  
**Home repo**: xeries  
**Input**: SDD bootstrap — describe the existing public surface of `ConditionalPermutationImportance`.

> **Status: Backfilled from implementation** — this spec describes the code as it
> exists on `main` at commit `858a498`. Redesigns require a follow-up spec.

## Summary

`ConditionalPermutationImportance` (cPFI) computes feature importance by
shuffling each feature **within homogeneous subgroups** and measuring the
resulting drop in model performance. Subgroups come from a `BasePartitioner`
— either user-defined (`ManualPartitioner`) or learned from the data via the
cs-PFI tree-based algorithm (`TreePartitioner`). Compared to vanilla PFI,
this preserves the conditional distribution of correlated features and so
gives less inflated, less misleading attributions for time-series problems
where lags, calendar features, and series-IDs are heavily intertwined.

## User Scenarios & Testing

### User Story 1 — Forecast-quality diagnostic on a global model (Priority: P1)

A user has a global LightGBM model trained across many series with strongly
correlated lag features. They want a faithful answer to "which features drive
forecast error".

**Independent Test**: train a model on a synthetic dataset with two correlated
lag features and one calendar feature; cPFI with `strategy="auto"` and
`metric="mse"` MUST rank the genuinely informative feature higher than the
correlated noise feature, and the result MUST satisfy
`isinstance(result, FeatureImportanceResult)`.

**Acceptance Scenarios**:

1. **Given** an explainer with `n_repeats=5`, **When** `explain` runs, **Then**
   `result.std` is a numpy array of length `len(features)` and every entry is
   non-negative.
2. **Given** `metric="r2"`, **When** `explain` runs, **Then** importance
   values are sign-corrected so that "higher = more important" regardless of
   whether the metric itself is loss-like or score-like.
3. **Given** `strategy="auto"` and no explicit `partitioner`, **When**
   `explain` runs, **Then** a fresh `TreePartitioner` is fitted per feature.

### User Story 2 — Per-series importance for hierarchical reporting (Priority: P2)

A user wants to know which features matter most for each individual series
(not just the global average), to feed a hierarchical report.

**Independent Test**: call
`explainer.explain_per_series(X, y, series_col="level")` (if exposed) OR feed
the result through `HierarchicalExplainer` (spec 006) — assert the per-series
DataFrame has one row per `(series, feature)` pair.

## Functional Requirements

### FR-003.1 Constructor

```python
ConditionalPermutationImportance(
    model: ModelProtocol,
    metric: MetricFunction | str = "mse",
    strategy: Literal["auto", "manual"] = "auto",
    partitioner: BasePartitioner | None = None,
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: int | None = None,
)
```

- Inherits `MetricBasedExplainer` (spec 001 FR-004).
- `metric="r2"` is supported and respects sklearn's "greater_is_better"
  convention.
- `n_jobs=-1` parallelises across features via `joblib.Parallel`.

### FR-003.2 `explain`

```python
def explain(
    self,
    X: pd.DataFrame,
    y: ArrayLike,
    features: list[str] | None = None,
    groups: GroupLabels | None = None,
    *args, **kwargs,
) -> FeatureImportanceResult: ...
```

- If `features is None`, defaults to `list(X.columns)`.
- For `strategy="auto"`, fits a `TreePartitioner` per feature and uses its
  leaf labels as conditional groups.
- For `strategy="manual"`, requires either `groups` or `self.partitioner`.
- For each feature × repeat, shuffles the feature within its group, calls
  `self.model.predict(X_shuf)`, scores against `y` using `self.metric`, and
  records `(score - baseline_score)` (with sign corrected for r2 / scoring
  metrics).
- Returns a `FeatureImportanceResult` whose `importances` are means across
  repeats and `std` are the corresponding standard deviations. The per-feature
  raw scores are stored in `permuted_scores` (key = feature name, value =
  list of `n_repeats` floats).
- `result.method == "conditional_permutation"`.

### FR-003.3 RNG discipline

- The explainer holds `self._rng = np.random.default_rng(random_state)` from
  `MetricBasedExplainer.__init__`. Per-feature workers MUST consume that
  generator (or a derived `SeedSequence`) — never `np.random.seed`.

## Public API surface

```python
from xeries.importance import ConditionalPermutationImportance
# Re-exported at top level:
from xeries import ConditionalPermutationImportance
```

Source file: `src/xeries/importance/permutation.py`.

## Test coverage

- `tests/unit/test_permutation.py` — unit tests for the explain pipeline,
  metric resolution, group dispatch, and reproducibility under fixed
  `random_state`.
- `examples/conditional_permutation/` — runnable notebooks demonstrating the
  end-to-end usage on small datasets.

## Out of scope / future work

- The currently parked branch `refactor/importance-sklearn` proposes to
  delegate the unconditional-PFI path to
  `sklearn.inspection.permutation_importance` (the conditional path stays
  hand-rolled because sklearn has no equivalent). When that lands, this
  Backfilled spec MUST be updated — it is the canonical surface contract.
- A `ConditionalDropImportance` (drop-and-refit instead of permute) is on
  the long-term roadmap but not scheduled.

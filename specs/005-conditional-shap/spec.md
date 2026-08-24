# Feature Specification: Conditional SHAP (`xeries.importance.shap`)

**Feature Branch**: `feat/sdd-adoption` (backfill)  
**Created**: 2026-04-26  
**Status**: Backfilled from implementation  
**Home repo**: xeries  
**Input**: SDD bootstrap — describe the existing public surface of `ConditionalSHAP`.

> **Status: Backfilled from implementation** — this spec describes the code as it
> exists on `main` at commit `858a498`. Redesigns require a follow-up spec.

## Summary

`ConditionalSHAP` computes SHAP attribution values for multi-series
forecasting models. It auto-detects the appropriate SHAP estimator
(`TreeExplainer` for tree ensembles, `LinearExplainer` for linear models,
`KernelExplainer` as the model-agnostic fallback, `DeepExplainer` for
neural nets), and supports two background-data strategies — `"series"` (a
per-series cohort) or `"global"` (sampled from the full dataset). The result
is a `SHAPResult` with both global and per-series aggregations.

## User Scenarios & Testing

### User Story 1 — Fast SHAP on a tree-based global forecaster (Priority: P1)

A user fits LightGBM / XGBoost / RandomForest on many series and wants SHAP
values without hand-picking the explainer type.

**Independent Test**:

```python
explainer = ConditionalSHAP(lgb_model, X_train, series_col="level")
result = explainer.explain(X_test)
assert isinstance(result, SHAPResult)
assert result.shap_values.shape == (len(X_test), X_test.shape[1])
```

**Acceptance Scenarios**:

1. **Given** a tree-based model and `explainer_type="auto"`, **When**
   `explain` is called, **Then** `TreeExplainer` is selected and batch SHAP
   is computed without sampling background data.
2. **Given** a non-tree model, **When** `explain` is called, **Then**
   `KernelExplainer` is selected and `n_background_samples` rows are sampled
   per series (or globally, depending on `background_strategy`).

### User Story 2 — Per-series cohort attributions (Priority: P1)

A user wants separate SHAP attributions per series, e.g. to feed
`HierarchicalExplainer` (spec 006) or to drill into a specific series.

**Independent Test**:

```python
results = explainer.explain_per_series(X_test)
assert set(results.keys()) == set(X_test.index.get_level_values("level").unique())
for series_id, r in results.items():
    assert isinstance(r, SHAPResult)
```

**Acceptance Scenarios**:

1. **Given** `background_strategy="series"`, **When** `explain` runs, **Then**
   each row's SHAP values are computed against a background sampled only
   from that row's series.
2. **Given** an explainer with no `series_col` in `X`, **When** `explain` is
   called, **Then** a clear `KeyError` is raised.

## Functional Requirements

### FR-005.1 Constructor

```python
ConditionalSHAP(
    model: ModelProtocol,
    background_data: pd.DataFrame,
    series_col: str = "level",
    n_background_samples: int = 100,
    explainer_type: Literal["tree", "linear", "kernel", "deep", "auto"] = "auto",
    explainer: Any = None,
    background_strategy: Literal["series", "global"] = "series",
    random_state: int | None = None,
)
```

- Inherits `AttributionExplainer` (spec 001 FR-005).
- If the optional `explainer` arg is supplied (a pre-built SHAP explainer),
  `explainer_type` is ignored.
- The constructor calls `_initialize()` which imports `shap` lazily and
  raises an `ImportError` with installation guidance if the library is not
  available.

### FR-005.2 Auto-detection rule

`explainer_type="auto"` MUST resolve as follows:

| Model class signature | Selected explainer |
| --- | --- |
| Has `tree_` / `booster_` / `feature_importances_` and a tree-like API | `TreeExplainer` (batch capable, no background) |
| Linear scikit-learn estimator | `LinearExplainer` |
| Has a `predict_proba` and a `keras` / `torch` module signature | `DeepExplainer` |
| Anything else | `KernelExplainer` with sampled background |

The exact detection logic lives in `ConditionalSHAP._initialize()`; this
table is the binding contract.

### FR-005.3 Background strategy

- `"series"`: per-series backgrounds. The explainer samples
  `n_background_samples` rows from each series in `background_data` at fit
  time and caches them in `self._series_backgrounds`.
- `"global"`: a single random sample of `n_background_samples` rows from the
  full `background_data` is cached in `self._global_background`.
- For `TreeExplainer` (tree path), no background is required — the strategy
  flag is recorded but ignored.

### FR-005.4 `explain` and `explain_per_series`

- `explain(X)` returns a `SHAPResult` with `shap_values`, `base_values`,
  `feature_names`, `data=X`, and `series_ids` filled in from `X[series_col]`
  (or the matching index level).
- `explain_per_series(X)` returns a `dict[series_id, SHAPResult]`.
- Both methods delegate to the underlying SHAP explainer in batch when
  `_is_batch_capable` is true (tree path), otherwise they iterate per row.

### FR-005.5 Determinism

- Background sampling uses `self._rng` (a `numpy.random.Generator`).
- Repeated calls to `explain` with the same `random_state` and the same
  inputs MUST produce identical SHAP values for kernel and deep paths
  (tree path is deterministic by construction).

## Public API surface

```python
from xeries.importance import ConditionalSHAP
from xeries.core import SHAPResult  # re-exported for typing
# Re-exported at top level:
from xeries import ConditionalSHAP
```

Source file: `src/xeries/importance/shap.py`.

## Test coverage

- `tests/unit/test_shap.py` — auto-detection, background sampling, batch
  vs. per-series modes, and reproducibility under fixed `random_state`.

## Out of scope / future work

- `ConditionalSHAPIQ` (any-order Shapley **interactions**) is on the
  roadmap as `specs/008-shapiq-explainer`. It will share the auto-detection
  philosophy of this spec but produce a `SHAPIQResult` (interaction tensor)
  rather than a flat `SHAPResult`. The two MUST coexist; the new spec will
  not redesign this one.
- The `ExplainerType = "deep"` path is wired but lightly tested. A follow-up
  spec MAY harden it once a real Keras/Torch use-case appears.

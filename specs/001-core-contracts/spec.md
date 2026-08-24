# Feature Specification: Core contracts (`xeries.core`)

**Feature Branch**: `feat/sdd-adoption` (backfill)  
**Created**: 2026-04-26  
**Status**: Backfilled from implementation  
**Home repo**: xeries  
**Input**: SDD bootstrap — describe the existing public surface of `xeries.core`.

> **Status: Backfilled from implementation** — this spec describes the code as it
> exists on `main` at commit `858a498`. Redesigns require a follow-up spec.

## Summary

`xeries.core` defines the **abstract contracts** that every other subsystem in
the library implements: a `ModelProtocol` for predictors, abstract base classes
for explainers and partitioners, and the result dataclasses returned from
explanations. Nothing in this module computes anything by itself — its purpose
is to be the shared interface that `xeries.partitioners`, `xeries.importance`,
`xeries.adapters`, and `xeries.hierarchy` all conform to.

## User Scenarios & Testing

### User Story 1 — A library author implements a new explainer (Priority: P1)

A contributor adding a new feature-importance method (e.g. SHAP-IQ) needs to
plug into the existing pipeline without breaking type-checks or tests.

**Independent Test**: write a stub class
`class FooExplainer(MetricBasedExplainer)` and verify it inherits a working
`__init__`, the `_resolve_metric` helper, and the `explain` abstract method
slot — confirmed by `ty check src` passing and `pytest -k foo` running the
new contract test.

**Acceptance Scenarios**:

1. **Given** a class derived from `BaseExplainer`, **When** it omits an
   `explain` implementation, **Then** instantiation raises `TypeError`
   (Python's normal abstract-method enforcement).
2. **Given** a class derived from `MetricBasedExplainer`, **When** it is
   instantiated with `metric="r2"`, **Then** `self.metric` resolves to a
   callable that wraps `sklearn.metrics.r2_score`.
3. **Given** a class derived from `MetricBasedExplainer`, **When** it is
   instantiated with `metric="not-a-metric"`, **Then** a `ValueError` is
   raised listing the supported names.

### User Story 2 — A user inspects results from any explainer uniformly (Priority: P1)

The user should not need to know which explainer produced a result — they
should be able to call `result.to_dataframe()` regardless of whether it came
from a permutation method, a SHAP method, or a causal method.

**Independent Test**: assert that
`isinstance(result, BaseResult)` for every public explainer's return type and
that all of `FeatureImportanceResult`, `SHAPResult`, `CausalResult`,
`HierarchicalResult` expose a `to_dataframe()` method.

## Functional Requirements (the contract)

### FR-001 `ModelProtocol`

- A `runtime_checkable` Protocol exposing `predict(X) -> ndarray | pd.Series`.
- Any object with a compatible `predict` method MUST be accepted by every
  explainer in the library — no isinstance checks against concrete classes.

### FR-002 `BasePartitioner`

- Abstract base class with `fit(X, feature) -> Self` and
  `get_groups(X) -> NDArray[np.intp]`.
- Provides a concrete `fit_get_groups(X, feature)` convenience that chains
  the two.
- Subclasses MUST return integer group labels with the same length as `X`.

### FR-003 `BaseExplainer`

- Abstract base class with `explain(X, *args, **kwargs) -> BaseResult`.

### FR-004 `MetricBasedExplainer`

- Concrete (but still abstract on `explain`) base for permutation-style
  explainers.
- `__init__` accepts `(model, metric: MetricFunction | str = "mse",
  random_state: int | None = None)`.
- `_resolve_metric` MUST recognise the strings `"mse"`, `"mae"`, `"rmse"`,
  `"r2"` and accept any callable `(y_true, y_pred) -> float`.
- Stores `self._rng = np.random.default_rng(random_state)` — explicit
  generator, never global `np.random.seed`.

### FR-005 `AttributionExplainer`

- Concrete base for SHAP / SHAP-IQ style explainers.
- `__init__(model, background_data: pd.DataFrame, random_state)`.
- Same `_rng` discipline.

### FR-006 `CausalExplainer`

- Concrete base for DAG / structural-causal explainers.
- Holds `treatment_features`, `causal_graph`, `series_col`.

### FR-007 Result containers

The following dataclasses live in `xeries.core.types` and inherit
`BaseResult`:

- `FeatureImportanceResult(feature_names, importances, std=None,
  baseline_score=0.0, permuted_scores={}, method="permutation",
  n_repeats=1)` — `.to_dataframe()` returns rows sorted by importance desc.
- `SHAPResult(shap_values, base_values, feature_names, data, series_ids=None)`
  — `.to_dataframe()`, `.mean_abs_shap()`, `.mean_abs_shap_by_series()`.
- `RefutationResult(method, original_effect, refuted_effect, p_value, passed)`.
- `CausalResult(feature_names, treatment_effects, confidence_intervals,
  p_values, causal_graph, estimator_name, refutation)` —
  `.to_dataframe()`, `.significant_features(alpha=0.05)`.

### FR-008 Type aliases

- `ArrayLike = np.ndarray | pd.Series | pd.DataFrame`
- `GroupLabels = np.ndarray | pd.Series | list[Any]`
- `MetricFunction = Callable[[ArrayLike, ArrayLike], float | int]`

## Public API surface

Re-exported from `xeries.core.__init__`:

```python
from xeries.core import (
    ArrayLike,
    BaseExplainer,
    BasePartitioner,
    FeatureImportanceResult,
    GroupLabels,
    ModelProtocol,
)
```

Source files: `src/xeries/core/base.py`, `src/xeries/core/types.py`.

## Test coverage

- Implicit, via every concrete explainer's test suite — `BasePartitioner` is
  exercised by `tests/unit/test_partitioners.py`, `MetricBasedExplainer` by
  `tests/unit/test_permutation.py`, `AttributionExplainer` by
  `tests/unit/test_shap.py`.
- No dedicated `tests/unit/test_core.py` exists at backfill time. A follow-up
  spec MAY add direct contract tests (e.g. assert `BaseExplainer` cannot be
  instantiated, assert `MetricBasedExplainer._resolve_metric` raises on
  unknown strings).

## Out of scope / future work

- The commented-out exports in `xeries.importance.__init__` (`CausalFeatureImportance`,
  `ConditionalDropImportance`, `ConditionalSHAPIQ`, `SHAPIQResult`) are NOT
  yet part of `xeries.core`'s contract. When they land, they will require new
  specs — `008-shapiq-explainer` is already on the roadmap.
- The `RefutationResult` and `CausalResult` dataclasses exist as forward
  scaffolding for a `CausalFeatureImportance` explainer that has not been
  implemented yet. Their public surface is therefore pre-Backfilled and may
  change without a redesign spec, until a `CausalFeatureImportance` ships.

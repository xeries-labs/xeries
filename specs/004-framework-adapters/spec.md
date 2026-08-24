# Feature Specification: Framework adapters (`xeries.adapters`)

**Feature Branch**: `feat/sdd-adoption` (backfill)  
**Created**: 2026-04-26  
**Status**: Backfilled from implementation  
**Home repo**: xeries  
**Input**: SDD bootstrap — describe the existing public surface of `xeries.adapters`.

> **Status: Backfilled from implementation** — this spec describes the code as it
> exists on `main` at commit `858a498`. Redesigns require a follow-up spec.

## Summary

`xeries` is framework-agnostic at its core (any object satisfying
`ModelProtocol` works), but real-world forecasters (skforecast, sklearn,
darts, …) ship their own training-data conventions, prediction signatures,
and series-ID columns. The **adapter layer** wraps those frameworks behind a
single `BaseAdapter` interface so the explainers can be reused unchanged.

Two concrete adapters ship; a Darts adapter is planned but not yet enabled
(see `xeries.adapters.__init__` for the commented-out import).

## User Scenarios & Testing

### User Story 1 — Plug a fitted skforecast forecaster into cPFI (Priority: P1)

A user has trained a `ForecasterRecursiveMultiSeries` and wants conditional
permutation importance on its lag features.

**Independent Test**:

```python
adapter = from_skforecast(forecaster, series=train_series)
X, y = adapter.get_training_data()
explainer = ConditionalPermutationImportance(model=adapter, metric="mse")
result = explainer.explain(X, y)
assert result.feature_names == adapter.get_feature_names()
```

**Acceptance Scenarios**:

1. **Given** a fitted skforecast forecaster, **When**
   `adapter.get_training_data()` is called, **Then** the returned `(X, y)`
   has the same lag-engineering as skforecast's internal call to `fit`.
2. **Given** the adapter, **When** `predict(X)` is called on a single
   row, **Then** the prediction matches `forecaster.regressor_.predict(X)`
   bit-for-bit.

### User Story 2 — Use a generic sklearn estimator (Priority: P1)

A user has a sklearn `Pipeline` and wants the same explainer surface without
hand-rolling features.

**Independent Test**: `SklearnAdapter(pipeline, X_train, y_train)` exposes
`get_training_data` returning the original `(X_train, y_train)` and `predict`
delegating to `pipeline.predict`.

## Functional Requirements

### FR-004.1 `BaseAdapter` (abstract)

```python
class BaseAdapter(ABC):
    @abstractmethod
    def get_training_data(self, *args, **kwargs) -> tuple[pd.DataFrame, pd.Series]: ...
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> NDArray[Any]: ...
    @abstractmethod
    def get_feature_names(self) -> list[str]: ...
    @abstractmethod
    def get_series_column(self) -> str: ...
```

- Any adapter MUST be usable as a `ModelProtocol` (i.e. expose `predict`).
- `get_series_column()` returns the column name (or index level name)
  containing series IDs — the explainers use it to set `series_col`.

### FR-004.2 `SklearnAdapter`

- Wraps any sklearn-compatible estimator (anything with `.predict`).
- `get_training_data(self)` returns the `(X, y)` provided at construction
  time.
- `get_feature_names()` returns `list(X.columns)`.
- `get_series_column()` returns the configured `series_col` (default
  `"level"`).

### FR-004.3 `SkforecastAdapter` and `from_skforecast`

- Wraps an skforecast `ForecasterRecursive*` family object.
- `get_training_data(series=None, **kwargs)` rebuilds the lagged training
  matrix with the **same** transformer, lags, and exogenous feature
  configuration as the underlying forecaster's call to `fit`. Callers MUST
  pass the original `series` object back in.
- `predict(X)` delegates to `forecaster.regressor_.predict(X)`.
- `from_skforecast(forecaster, series=...)` is a convenience constructor that
  validates the forecaster type and wires up `series_col` from the
  forecaster's internal level column (skforecast 0.21+ uses
  `_level_skforecast`; legacy uses `level`).

### FR-004.4 Compatibility surface

- Supported skforecast: ≥ 0.14 (legacy `level` column path) and ≥ 0.21
  (preferred `_level_skforecast` path).
- Supported sklearn: any estimator that conforms to the regressor API.

## Public API surface

```python
from xeries.adapters import (
    BaseAdapter,
    SklearnAdapter,
    SkforecastAdapter,
    from_skforecast,
)
# Re-exported at top level:
from xeries import (
    BaseAdapter,
    SklearnAdapter,
    SkforecastAdapter,
    from_skforecast,
)
```

Source files: `src/xeries/adapters/base.py`, `src/xeries/adapters/sklearn.py`,
`src/xeries/adapters/skforecast.py`.

## Test coverage

- `tests/integration/test_skforecast.py` — end-to-end test that fits a real
  skforecast forecaster, builds the adapter, and runs cPFI on top.
- The `SklearnAdapter` is exercised via the cPFI / SHAP unit tests, which all
  use simple sklearn regressors.

## Out of scope / future work

- `DartsAdapter` (commented out in `xeries.adapters.__init__`) is planned
  but not yet enabled. When it lands it MUST satisfy this spec's
  `BaseAdapter` contract.
- Multi-output models (e.g. direct multi-step forecasters) are currently
  treated as `predict` returning shape `(n,)`. A future spec may extend
  `BaseAdapter.predict` to `NDArray[..., n_horizons]`.

# Getting Started

This guide will help you install xeries and understand the core concepts.

## Installation

### Using pip

```bash
pip install xeries
```

### Using UV

```bash
uv add xeries
```

### Optional Dependencies

For skforecast integration:

```bash
pip install xeries[skforecast]
```

For development:

```bash
uv sync --dev
```

## Core Concepts

### Partitioners

Partitioners define how data is grouped for conditional permutation. xeries provides two main approaches:

#### Manual Partitioner

Use when you have domain knowledge about how series should be grouped:

```python
from xeries import ManualPartitioner

mapping = {
    'store_001': 'urban',
    'store_002': 'suburban',
    'store_003': 'urban',
}
partitioner = ManualPartitioner(mapping, series_col='store_id')
```

With **skforecast** (0.21+), pass `series_col=adapter.get_series_column()`. If that resolves to `_level_skforecast`, use integer keys `0, 1, …` in `mapping` in the same order as `forecaster.series_names_in_`.

#### Tree Partitioner

Automatically learns subgroups using a decision tree:

```python
from xeries import TreePartitioner

partitioner = TreePartitioner(
    max_depth=4,
    min_samples_leaf=0.05,
    series_col=None,  # auto: "_level_skforecast" (skforecast 0.21+) or "level"
)
```

### Importance Methods

#### Conditional Permutation Importance

```python
from xeries import ConditionalPermutationImportance

explainer = ConditionalPermutationImportance(
    model=model,
    metric='mse',
    strategy='auto',  # or 'manual'
    n_repeats=5,
)

result = explainer.explain(X, y, features=['lag_1', 'lag_2'])
df = result.to_dataframe()
```

## Working with skforecast

xeries integrates seamlessly with skforecast:

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries
from xeries.adapters.skforecast import SkforecastAdapter, from_skforecast
from xeries import ConditionalPermutationImportance

# Train your forecaster (skforecast 0.21+)
forecaster = ForecasterRecursiveMultiSeries(estimator=model, lags=24)
forecaster.fit(series=data)

# Pass the same `series` as fit(series=...) (required by skforecast.create_train_X_y)
adapter = from_skforecast(forecaster, series=data)

# Get training data
X, y = adapter.get_training_data()

# Compute importance
explainer = ConditionalPermutationImportance(model=adapter, metric='mse')
result = explainer.explain(X, y)
```

## Visualization

xeries includes plotting utilities:

```python
from xeries.visualization import plot_importance_bar

fig, ax = plot_importance_bar(result, max_features=10)
```

## Planned Methods

The following methods are planned for future releases:

- SHAP-IQ
- Feature Dropping
- Causal Feature Importance
- Darts adapter

## Next Steps

- Follow the [Quickstart Tutorial](tutorials/quickstart.md) for a complete example
- Browse [Examples](examples.md) for executed notebooks
- Explore the [API Reference](api/reference.md) for detailed documentation

# Xeries

**Time Series Explainability & Conditional Permutation Feature Importance**

A Python library for generalized time series explainability in multi-time series forecasting, supporting feature importance, SHAP, and causal methods.

## Overview

When using global models for multi-time series forecasting, standard feature importance methods can produce misleading results because they fail to respect the conditional nature of series-dependent features (like lags and series identifiers). This leads to unrealistic data permutations and unreliable insights.

**xeries** addresses this challenge by providing a generalized approach:

- **Generic Explainability Architecture**: A unified foundation for multiple interpretation methodologies
- **Conditional Permutation Importance**: Permutes features only within meaningful subgroups
- **Tree-Based cs-PFI**: Automatically learns homogeneous subgroups using decision trees
- **Manual Grouping**: Use domain knowledge to define custom permutation groups
- **Planned Methods**: Conditional SHAP, SHAP-IQ, feature dropping, and causal feature importance are planned for future releases
- **Framework Integration**: Works with skforecast

## Installation

```bash
pip install xeries
```

Or with UV:

```bash
uv add xeries
```

For skforecast integration:

```bash
pip install xeries[skforecast]
```

## Quick Example

```python
import xeries
from sklearn.ensemble import RandomForestRegressor

# Assume you have a trained model and data
explainer = xeries.ConditionalPermutationImportance(
    model=model,
    metric='mse',
    strategy='auto'  # Uses tree-based cs-PFI
)

result = explainer.explain(X, y, features=['lag_1', 'lag_2', 'day_of_week'])
print(result.to_dataframe())
```

## Key Concepts

### Series-Dependent Features

In multi-series forecasting, many features are intrinsically tied to their series:

- **Autoregressive lags**: `lag_1` for Product A is meaningless for Product B
- **Rolling statistics**: A 7-day rolling mean is series-specific
- **Series identifiers**: Explicitly tell the model which series a row belongs to
- **Exogenous features**: May have different distributions across series (e.g., price, promotion)

### Why Conditional Importance?

Standard permutation importance shuffles feature values across the entire dataset, potentially pairing a lag from "Product A" with a target from "Product B". This creates nonsensical data and invalid importance scores.

**Conditional importance** constrains permutations within subgroups where the permutation makes sense, preserving the data's structural integrity.

## Next Steps

- [Getting Started](getting-started.md): Detailed setup and first steps
- [Quickstart Tutorial](tutorials/quickstart.md): End-to-end example with skforecast
- [API Reference](api/reference.md): Complete API documentation

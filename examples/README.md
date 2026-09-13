# Examples

This directory contains Jupyter notebooks demonstrating how to use xeries, organized by method type. The same notebooks are rendered (with saved outputs) on the [documentation site](https://xeries-labs.github.io/xeries/examples/).

## Folder Structure

```
examples/
├── quickstart/               # Getting started examples
├── conditional_permutation/  # Conditional permutation importance
├── hierarchical/             # Hierarchy + SHAP
└── integrations/             # Framework integrations
```

## Notebooks

### Quickstart

| Notebook | Description | Dependencies |
|----------|-------------|--------------|
| [01_basic_usage.ipynb](quickstart/01_basic_usage.ipynb) | Basic usage with synthetic data | Core xeries |

### Conditional Permutation Importance

| Notebook | Description | Dependencies |
|----------|-------------|--------------|
| [01_per_series_importance.ipynb](conditional_permutation/01_per_series_importance.ipynb) | Per-series feature importance analysis | Core xeries |
| [02_exogenous_features.ipynb](conditional_permutation/02_exogenous_features.ipynb) | Heterogeneous effects of exogenous variables | Core xeries |
| [03_per_series_exog.ipynb](conditional_permutation/03_per_series_exog.ipynb) | Per-series importance with exogenous features | `xeries[notebooks]` (LightGBM) |
| [04_tree_partitioner.ipynb](conditional_permutation/04_tree_partitioner.ipynb) | TreePartitioner for automatic subgroup discovery | `xeries[skforecast]` |

### Hierarchical

| Notebook | Description | Dependencies |
|----------|-------------|--------------|
| [01_hierarchical_demand_forecasting.ipynb](hierarchical/01_hierarchical_demand_forecasting.ipynb) | Hierarchical aggregation of explanations | `xeries[skforecast,notebooks]` |
| [02_per_series_shap.ipynb](hierarchical/02_per_series_shap.ipynb) | Per-series SHAP | `xeries[skforecast,notebooks]` |

### Integrations

| Notebook | Description | Dependencies |
|----------|-------------|--------------|
| [01_skforecast.ipynb](integrations/01_skforecast.ipynb) | Integration with skforecast | `xeries[skforecast]` |

## Running the Notebooks

From a clone of this repository:

```bash
uv sync --extra notebooks --extra skforecast
uv run jupyter notebook
```

With pip (published package):

```bash
pip install "xeries[skforecast,notebooks]"
```

## Testing Notebooks

```bash
# Structure tests (fast)
uv run pytest tests/test_notebooks.py -m "notebook and not slow"

# Execution tests (slower)
uv run pytest tests/test_notebooks.py -m "notebook and slow"
```

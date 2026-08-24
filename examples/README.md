# Examples

This directory contains Jupyter notebooks demonstrating how to use xeries, organized by method type.

## Folder Structure

```
examples/
├── quickstart/           # Getting started examples
├── conditional_permutation/  # Conditional permutation importance
└── integrations/         # Framework integrations
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
| [03_per_series_exog.ipynb](conditional_permutation/03_per_series_exog.ipynb) | Per-series importance with exogenous features | Core xeries, lightgbm |
| [04_tree_partitioner.ipynb](conditional_permutation/04_tree_partitioner.ipynb) | TreePartitioner for automatic subgroup discovery | `xeries[skforecast]` |

### Integrations

| Notebook | Description | Dependencies |
|----------|-------------|--------------|
| [01_skforecast.ipynb](integrations/01_skforecast.ipynb) | Integration with skforecast | `xeries[skforecast]` |


## Running the Notebooks

### Install Dependencies

```bash
# Install xeries with notebook support
pip install xeries[notebooks]

# Or with UV
uv add xeries --extra notebooks
```

For skforecast integration:

```bash
pip install xeries[skforecast,notebooks]
```

For LightGBM examples:

```bash
pip install lightgbm
```

### Launch Jupyter

```bash
jupyter notebook
```

Then open any notebook from the file browser.

## Testing Notebooks

Notebooks are automatically tested in CI to ensure they execute without errors.

To run notebook tests locally:

```bash
# Structure tests (fast)
pytest tests/test_notebooks.py -m "notebook and not slow"

# Execution tests (slower)
pytest tests/test_notebooks.py -m "notebook and slow"
```

# Examples

Runnable Jupyter notebooks live in the repository under [`examples/`](https://github.com/xeries-labs/xeries/tree/main/examples). This site renders those notebooks **with saved outputs** (tables and plots).

Install extras before running them locally:

```bash
# from a clone
uv sync --extra notebooks --extra skforecast --group docs
```

Or with pip:

```bash
pip install "xeries[skforecast,notebooks]"
```

## Quickstart

- [Basic usage](examples/quickstart/01_basic_usage.ipynb) — synthetic multi-series data and cs-PFI without a forecasting library.

## Conditional permutation importance

- [Per-series importance](examples/conditional_permutation/01_per_series_importance.ipynb)
- [Exogenous features](examples/conditional_permutation/02_exogenous_features.ipynb)
- [Per-series with exogenous features](examples/conditional_permutation/03_per_series_exog.ipynb) — requires LightGBM (`xeries[notebooks]`)
- [Tree partitioner](examples/conditional_permutation/04_tree_partitioner.ipynb) — requires `xeries[skforecast]`

## Hierarchical explanations

- [Hierarchical demand forecasting](examples/hierarchical/01_hierarchical_demand_forecasting.ipynb)
- [Per-series SHAP](examples/hierarchical/02_per_series_shap.ipynb)

## Integrations

- [skforecast](examples/integrations/01_skforecast.ipynb)

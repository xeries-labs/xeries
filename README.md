# xeries

[![CI](https://github.com/xeries-labs/xeries/actions/workflows/ci.yml/badge.svg)](https://github.com/xeries-labs/xeries/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/xeries.svg)](https://badge.fury.io/py/xeries)
[![Python versions](https://img.shields.io/pypi/pyversions/xeries.svg)](https://pypi.org/project/xeries/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19482748.svg)](https://doi.org/10.5281/zenodo.19482748)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Time Series eXplainability (XAI) for Forecasting**

A comprehensive Python library for explainability and interpretability in multi-time series forecasting. **xeries** provides multiple explanation methods—including conditional permutation importance, SHAP, feature dropping, and causal analysis—through a unified API for understanding forecast decisions.

## Why xeries?

Explaining and interpreting multi-time series forecasts is challenging:

1. **Standard methods fail on series-dependent features**: Permutation importance and SHAP can create invalid data by pairing a lag from one series with a target from another.

2. **No unified explainability framework**: Most libraries offer a single method (e.g., SHAP only) with limited support for time-series-specific analysis.

3. **Lack of temporal and comparative insights**: Understanding performance across time windows, detecting significance, and comparing methods are difficult without custom tools.

**xeries** addresses these gaps with:
- **Conditional explanations** that respect data structure
- **A focused CPI workflow** for current production use
- **A roadmap for future explanation methods**
- **A simple API** for current cs-PFI workflows

## Features

### 📊 Current Capability

- **Conditional Permutation Importance (cs-PFI)**: Permute features only within meaningful subgroups
  - Auto-discover groups using decision trees
  - Define custom groups based on domain knowledge

### 🛣️ Planned Roadmap

- **Conditional SHAP**: Planned for a future release
- **SHAP-IQ**: Planned for a future release
- **Feature Dropping**: Planned for a future release
- **Causal Feature Importance**: Planned for a future release

### 📡 Framework Adapters

- **scikit-learn**: Direct support for sklearn estimators
- **skforecast**: Seamless integration with multi-series forecasters (0.21+)
- **Custom Models**: Wrap any forecaster with the BaseAdapter

### 📈 Visualization

Ready-to-use plotting utilities for feature importance and temporal analysis.
- **Publication-Ready Plots**:
  - Feature importance bar charts
  - Heatmaps comparing multiple methods or conditions
  - Per-series and method-comparison plots
- **Jupyter Integration**: Works seamlessly in notebooks

> **Dashboard / HTML reports** (Jinja2-templated reports, interactive widgets, method-comparison
> dashboards) moved to the sibling `xeries-labs/xeries-dashboard` repo per governance v1.2.0.
> See roadmap row `specs/013-dashboard`.

## Installation

```bash
pip install xeries
```

With UV:

```bash
uv add xeries
```

For skforecast integration:

```bash
pip install xeries[skforecast]
```

## Supported Explanation Methods

| Method | Status | Use Case | Notes |
|--------|--------|----------|-------|
| **Conditional Permutation Importance** | Available | Default choice; fast & interpretable | Auto/manual grouping with tree-based or manual partitioning |
| **Conditional SHAP** | Planned | Local & global explanations | Future release |
| **SHAP-IQ** | Planned | Feature interactions | Future release |
| **Feature Dropping** | Planned | Complementary to importance | Future release |
| **Causal Feature Importance** | Planned | Treatment effects | Future release |

## Quick Start

### Conditional Permutation Importance (Default)

```python
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

from xeries import ConditionalPermutationImportance
from xeries.adapters.skforecast import from_skforecast

# Train your multi-series forecaster (skforecast 0.21+)
forecaster = ForecasterRecursiveMultiSeries(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    lags=24,
)
forecaster.fit(series=your_data)

# Same `series` as fit() is required for create_train_X_y (pass here or to get_training_data)
adapter = from_skforecast(forecaster, series=your_data)
X, y = adapter.get_training_data()

# Compute conditional importance (automatic tree-based cs-PFI)
explainer = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='auto',
    n_repeats=5,
    random_state=42,
)

result = explainer.explain(X, y, features=['lag_1', 'lag_2', 'lag_3'])
print(result.to_dataframe())
```

### Using Manual Groups

```python
from xeries import ManualPartitioner, ConditionalPermutationImportance

# Domain groups: with skforecast 0.21+ wide data, series are ordinal-encoded in X.
# Map integers 0,1,... in the same order as forecaster.series_names_in_ (see adapter.forecaster).
mapping = {
    0: 'urban',
    1: 'suburban',
    2: 'urban',
}

partitioner = ManualPartitioner(mapping, series_col=adapter.get_series_column())

explainer = ConditionalPermutationImportance(
    model=adapter,
    metric='mse',
    strategy='manual',
    partitioner=partitioner,
)

result = explainer.explain(X, y)
```

### Visualization

```python
from xeries.visualization import plot_importance_bar, plot_importance_heatmap

# Bar chart
fig, ax = plot_importance_bar(result, max_features=10)

# Heatmap comparing multiple cs-PFI configurations
results = {'Auto': result_auto, 'Manual': result_manual}
fig, ax = plot_importance_heatmap(results)
```

## Planned Methods

The following methods are planned for future releases and are not part of the current release:

- Conditional SHAP
- SHAP-IQ
- Feature Dropping
- Causal Feature Importance

## Documentation

Full documentation is available at [https://thec0dewriter.github.io/xeries](https://thec0dewriter.github.io/xeries)

## Spec-Driven Development

`xeries` is developed under the **[xeries-labs](https://github.com/xeries-labs)** Spec-Driven Development (SDD) program, governed by [`xeries-labs/xeries-governance`](https://github.com/xeries-labs/xeries-governance) (mounted in this repo as a git submodule at `.specify/`). Before contributing or asking an AI agent to work on this codebase, please read:

- [`AGENTS.md`](./AGENTS.md) — the entry point for human and AI contributors.
- `.specify/memory/constitution.md` — binding program rules (after `git submodule update --init`).
- `.specify/memory/roadmap.md` — the program-wide roadmap (xeries + xeries-bench).
- [`docs/contributing-sdd.md`](./docs/contributing-sdd.md) — full SDD workflow walkthrough (also rendered in the docs site).

Backfilled specifications for the existing public surface live under [`specs/001-…`](./specs/) through [`specs/007-…`](./specs/). New features (e.g. `008-shapiq-explainer`) start with a `spec.md` from `.specify/templates/spec-template.md` and follow the constitution's gates.

## Development

Clone the repository (with the governance submodule):

```bash
git clone --recurse-submodules https://github.com/xeries-labs/xeries.git
cd xeries
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Install with development dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check src tests
uv run ruff format src tests
```

Type checking:

```bash
uv run ty check src
```

Build documentation:

```bash
uv sync --group docs
uv run mkdocs serve
```

## Contributing

Contributions are welcome under the project's Spec-Driven Development workflow. **Read [`AGENTS.md`](./AGENTS.md) and [`docs/contributing-sdd.md`](./docs/contributing-sdd.md) before starting** — they describe the constitution gates, the spec → plan → tasks flow, and the Repo-scope rules (e.g. why `lightgbm` belongs in `xeries-bench`, not here).

Short version:

1. Fork the repository.
2. Open or update a spec under `specs/NNN-<slug>/` (use `.specify/templates/spec-template.md`). For mechanical refactors that preserve the public surface, cite the affected Backfilled spec(s) in the commit message instead of authoring a new one.
3. For new specs, fill out a `plan.md` and pass the **Constitution Check** gate before writing code.
4. Create a feature branch (`git checkout -b feature/<short-name>`).
5. Add tests **before** the implementation commit when you introduce a new public symbol (Principle III — Test-first for new contracts).
6. Run `uv run ruff check src tests`, `uv run ty check src`, `uv run pytest`, `uv run mkdocs build --strict` locally before pushing.
7. Open a Pull Request — CI replays the same gates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{xeries,
  title = {xeries: Time Series eXplainability for Forecasting},
  author = {Kuti-Kreszács, Mátyás},
  year = {2026},
  doi = {10.5281/zenodo.19482748},
  publisher = {Zenodo},
  url = {https://github.com/xeries-labs/xeries},
}
```

### Related Publications

This project is informed by the following techniques and references:

- **Conditional Subgroup Permutation Feature Importance** (cs-PFI)
- **SHAP** (SHapley Additive exPlanations) — [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- **SHAP Interaction Values** — [Janzing et al., 2020](https://arxiv.org/abs/1908.08474)
- **Controlled Feature Dropping** — Model modification for importance estimation
- **Causal Inference** using **DoWhy** and **EconML** frameworks
- **skforecast** — [Multi-series forecasting framework](https://skforecast.org/)

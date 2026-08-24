# Architecture Overview

This section provides a comprehensive overview of the xeries library architecture, designed for explainability in multi-time series forecasting.

## High-Level Architecture

```mermaid
graph TB
    subgraph core [Core Module]
        base[base.py<br/>BaseExplainer, BasePartitioner]
        types[types.py<br/>SHAPResult, FeatureImportanceResult]
    end

    subgraph importance [Importance Module]
        perm[permutation.py<br/>ConditionalPermutationImportance]
        shap[shap.py<br/>ConditionalSHAP]
    end

    subgraph hierarchy [Hierarchy Module]
        defn[definition.py<br/>HierarchyDefinition]
        agg[aggregator.py<br/>HierarchicalAggregator]
        hexpl[explainer.py<br/>HierarchicalExplainer]
        htypes[types.py<br/>HierarchicalResult]
    end

    subgraph partitioners [Partitioners Module]
        tree[tree.py<br/>TreePartitioner]
        manual[manual.py<br/>ManualPartitioner]
    end

    subgraph visualization [Visualization Module]
        plots[plots.py<br/>Standard plots]
        hplots[hierarchy_plots.py<br/>Hierarchy plots]
    end

    subgraph adapters [Adapters Module]
        sklearn[sklearn.py<br/>SklearnAdapter]
        skforecast[skforecast.py<br/>SkforecastAdapter]
    end

    base --> perm
    base --> shap
    types --> perm
    types --> shap
    perm --> hexpl
    shap --> hexpl
    defn --> agg
    agg --> hexpl
    htypes --> hexpl
```

## Design Principles

### 1. Unified Explainer Interface

All explainers inherit from `BaseExplainer` and implement the `explain()` method:

```python
class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, X: pd.DataFrame, *args, **kwargs) -> BaseResult:
        ...
```

### 2. Composable Architecture

Components can be composed together:

```python
# Compose explainer with hierarchy
base_explainer = ConditionalSHAP(model, X_train, series_col='level')
hierarchical = HierarchicalExplainer(base_explainer, hierarchy)
result = hierarchical.explain(X_test)
```

### 3. Result-Oriented Design

All methods return typed result objects with utility methods:

- `SHAPResult` - SHAP values with `mean_abs_shap()`, `to_dataframe()`, `mean_abs_shap_by_series()`
- `FeatureImportanceResult` - Permutation importance with `to_dataframe()`
- `HierarchicalResult` - Multi-level aggregated results with `get_level_df()`, `get_global()`

## Module Documentation

- [Core Module](core.md) - Base classes and type definitions
- [Importance Module](importance.md) - Feature importance methods
- [Hierarchy Module](hierarchy.md) - Hierarchical aggregation
- [Partitioners Module](partitioners.md) - Data partitioning strategies
- [Visualization Module](visualization.md) - Plotting utilities
- [Adapters Module](adapters.md) - Framework integrations

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        Model[Model<br/>trained]
        Data[X DataFrame]
        BGData[Background Data]
    end

    subgraph Explainers
        SHAP[ConditionalSHAP]
        PFI[ConditionalPermutation<br/>Importance]
    end

    subgraph Results
        SHAPRes[SHAPResult]
        PFIRes[FeatureImportance<br/>Result]
    end

    subgraph Hierarchy [Hierarchical Processing]
        HierDef[HierarchyDefinition]
        HierExpl[HierarchicalExplainer]
        HierRes[HierarchicalResult]
    end

    subgraph Viz [Visualization]
        Plots[plot_importance_bar<br/>plot_shap_summary]
        HPlots[plot_hierarchy_summary<br/>plot_hierarchy_violin]
    end

    Model --> SHAP
    Model --> PFI
    Data --> SHAP
    Data --> PFI
    BGData --> SHAP

    SHAP --> SHAPRes
    PFI --> PFIRes

    SHAPRes --> HierExpl
    PFIRes --> HierExpl
    HierDef --> HierExpl
    HierExpl --> HierRes

    SHAPRes --> Plots
    PFIRes --> Plots
    HierRes --> HPlots
```

## Workflow Examples

### Basic Workflow

```python
from xeries import ConditionalSHAP

# 1. Create explainer
explainer = ConditionalSHAP(model, X_train, series_col='level')

# 2. Compute explanations
result = explainer.explain(X_test)

# 3. Analyze results
print(result.mean_abs_shap())
```

### Hierarchical Workflow

```python
from xeries import ConditionalSHAP
from xeries.hierarchy import HierarchyDefinition, HierarchicalExplainer
from xeries.visualization import plot_hierarchy_summary

# 1. Define hierarchy
hierarchy = HierarchyDefinition(
    levels=['state', 'store'],
    columns=['state_id', 'store_id']
)

# 2. Create hierarchical explainer
base = ConditionalSHAP(model, X_train, series_col='level')
explainer = HierarchicalExplainer(base, hierarchy)

# 3. Compute hierarchical results
result = explainer.explain(X_test, include_raw=True)

# 4. Visualize
plot_hierarchy_summary(result)
```

### Per-Series Workflow

```python
from xeries import ConditionalPermutationImportance, ConditionalSHAP

# Permutation importance per series
pfi_explainer = ConditionalPermutationImportance(model, metric='mse')
pfi_results = pfi_explainer.explain_per_series(X, y, series_col='level')

# SHAP values per series
shap_explainer = ConditionalSHAP(model, X_train, series_col='level')
shap_results = shap_explainer.explain_per_series(X_test)
```

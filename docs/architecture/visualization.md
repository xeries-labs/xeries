# Visualization Module

The visualization module provides plotting utilities for feature importance results.

## Location

`xeries/visualization/`

## Architecture

```mermaid
classDiagram
    class StandardPlots {
        <<plots.py>>
        +plot_importance_bar(result, ...) Figure, Axes
        +plot_importance_comparison(results, ...) Figure, Axes
        +plot_importance_heatmap(results, ...) Figure, Axes
        +plot_importance_per_series(results, ...) Figure, Axes
        +plot_shap_bar(result, ...) Figure, Axes
        +plot_shap_summary(shap_values, ...) Figure, Axes
    }

    class HierarchyPlots {
        <<hierarchy_plots.py>>
        +plot_hierarchy_bar(result, ...) Figure, Axes
        +plot_hierarchy_violin(result, ...) Figure, Axes
        +plot_hierarchy_comparison(result, ...) Figure, Axes
        +plot_hierarchy_summary(result, ...) Figure, Axes
        +plot_hierarchy_tree(result, ...) Figure, Axes
        +plot_hierarchy_heatmap(result, ...) Figure, Axes
    }
```

## Standard Plots (`plots.py`)

### plot_importance_bar

Horizontal bar chart of feature importance.

```mermaid
flowchart LR
    Input[FeatureImportanceResult<br/>or SHAPResult]
    Plot[Horizontal Bar Chart]
    Input --> Plot
```

```python
from xeries.visualization import plot_importance_bar

fig, ax = plot_importance_bar(
    result,
    top_n=10,
    figsize=(10, 6),
    title="Feature Importance"
)
```

### plot_importance_comparison

Compare importance across multiple results.

```mermaid
flowchart LR
    Results["dict[name, Result]"]
    Plot[Grouped Bar Chart]
    Results --> Plot
```

```python
from xeries.visualization import plot_importance_comparison

results = {
    'Series A': result_a,
    'Series B': result_b,
}

fig, ax = plot_importance_comparison(
    results,
    top_n=5,
    figsize=(12, 6)
)
```

### plot_importance_heatmap

Heatmap of importance across series.

```mermaid
flowchart LR
    Results["dict[series, Result]"]
    Plot[Heatmap<br/>features x series]
    Results --> Plot
```

```python
from xeries.visualization import plot_importance_heatmap

per_series_results = explainer.explain_per_series(X, y, series_col='level')

fig, ax = plot_importance_heatmap(
    per_series_results,
    top_n=10,
    figsize=(12, 8)
)
```

### plot_shap_summary

SHAP summary plot (beeswarm style).

```python
from xeries.visualization import plot_shap_summary

fig, ax = plot_shap_summary(
    result.shap_values,
    result.data,
    result.feature_names,
    max_display=10
)
```

---

## Hierarchy Plots (`hierarchy_plots.py`)

### plot_hierarchy_bar

Bar chart for a specific hierarchy level and cohort.

```mermaid
flowchart LR
    HResult[HierarchicalResult]
    Level["level='state'"]
    Cohort["cohort='TX'"]
    Plot[Horizontal Bar Chart]
    
    HResult --> Plot
    Level --> Plot
    Cohort --> Plot
```

```python
from xeries.visualization import plot_hierarchy_bar

fig, ax = plot_hierarchy_bar(
    result,
    level='state',
    cohort='TX',
    top_n=10
)
```

### plot_hierarchy_violin

SHAP beeswarm/violin plot for a cohort.

```mermaid
flowchart LR
    HResult[HierarchicalResult<br/>with raw_values]
    Level["level='state'"]
    Cohort["cohort='TX'"]
    Plot[Beeswarm Plot<br/>colored by feature value]
    
    HResult --> Plot
    Level --> Plot
    Cohort --> Plot
```

```python
from xeries.visualization import plot_hierarchy_violin

# Requires include_raw=True when computing
result = explainer.explain(X, include_raw=True)

fig, ax = plot_hierarchy_violin(
    result,
    level='state',
    cohort='TX',
    top_n=10
)
```

### plot_hierarchy_summary

Grid of plots for all hierarchy levels.

```mermaid
flowchart TB
    HResult[HierarchicalResult]
    
    subgraph Grid [Output Grid]
        R1C1[Global: all]
        R2C1[State: TX]
        R2C2[State: WI]
        R3C1[Store: TX_S1]
        R3C2[Store: TX_S2]
        R3C3[Store: WI_S1]
    end
    
    HResult --> Grid
```

```python
from xeries.visualization import plot_hierarchy_summary

fig, axes = plot_hierarchy_summary(
    result,
    levels=None,  # All levels
    top_n=5,
    figsize=(15, 10)
)
```

### plot_hierarchy_comparison

Grouped bar chart comparing cohorts at a level.

```python
from xeries.visualization import plot_hierarchy_comparison

fig, ax = plot_hierarchy_comparison(
    result,
    level='state',
    top_n=5,
    figsize=(12, 6)
)
```

### plot_hierarchy_heatmap

Heatmap of importance across cohorts.

```python
from xeries.visualization import plot_hierarchy_heatmap

fig, ax = plot_hierarchy_heatmap(
    result,
    level='store',
    top_n=10,
    figsize=(12, 8)
)
```

## Usage Example

```python
from xeries import ConditionalSHAP
from xeries.hierarchy import HierarchyDefinition, HierarchicalExplainer
from xeries.visualization import (
    plot_hierarchy_summary,
    plot_hierarchy_bar,
    plot_hierarchy_violin,
    plot_hierarchy_comparison
)
import matplotlib.pyplot as plt

# Setup
hierarchy = HierarchyDefinition(
    levels=['state', 'store'],
    columns=['state_id', 'store_id']
)
base = ConditionalSHAP(model, X_train, series_col='level')
explainer = HierarchicalExplainer(base, hierarchy)

# Compute with raw values for violin plots
result = explainer.explain(X_test, include_raw=True)

# Summary grid
fig, axes = plot_hierarchy_summary(result)
plt.suptitle("Feature Importance Across Hierarchy")
plt.show()

# State comparison
fig, ax = plot_hierarchy_comparison(result, level='state', top_n=5)
plt.show()

# Violin plot for TX
fig, ax = plot_hierarchy_violin(result, level='state', cohort='TX')
plt.show()
```

## Plot Gallery

```mermaid
flowchart TB
    subgraph Standard [Standard Plots]
        ImpBar[plot_importance_bar<br/>Single result bar chart]
        ImpComp[plot_importance_comparison<br/>Multiple results grouped bars]
        ImpHeat[plot_importance_heatmap<br/>Features x Series heatmap]
        ShapSum[plot_shap_summary<br/>SHAP beeswarm plot]
    end

    subgraph Hierarchy [Hierarchy Plots]
        HBar[plot_hierarchy_bar<br/>Single cohort bar chart]
        HViolin[plot_hierarchy_violin<br/>Cohort beeswarm plot]
        HSummary[plot_hierarchy_summary<br/>Grid of all levels]
        HComp[plot_hierarchy_comparison<br/>Compare cohorts at level]
        HHeat[plot_hierarchy_heatmap<br/>Cohorts x Features heatmap]
    end
```

# Feature Specification: Visualization (`xeries.visualization`)

**Feature Branch**: `feat/sdd-adoption` (backfill)  
**Created**: 2026-04-26  
**Status**: Backfilled from implementation  
**Home repo**: xeries  
**Input**: SDD bootstrap — describe the existing public surface of `xeries.visualization`.

> **Status: Backfilled from implementation** — this spec describes the code as it
> exists on `main` at commit `858a498`. Redesigns require a follow-up spec.

## Summary

`xeries.visualization` provides a thin layer of **opinionated, reproducible**
plotting helpers on top of the result containers (`FeatureImportanceResult`,
`SHAPResult`, `HierarchicalResult`). The intent is to give analysts a one-line
path from result → plot, while keeping the underlying matplotlib axes
returned so users can compose them into bigger figures or save them in any
backend they prefer.

All plotting helpers are **pure presentation** — they do not recompute
attributions and they do not mutate their inputs.

## User Scenarios & Testing

### User Story 1 — One-line bar chart of importance ranking (Priority: P1)

A user wants the standard "ranked bar chart" of feature importances from a
`FeatureImportanceResult` or `SHAPResult`.

**Independent Test**:

```python
ax = plot_importance_bar(result)
assert ax is not None
assert isinstance(ax, plt.Axes)
```

**Acceptance Scenarios**:

1. **Given** a `FeatureImportanceResult`, **When** `plot_importance_bar(r)`
   is called, **Then** the returned `Axes` has one bar per feature, sorted
   by importance descending.
2. **Given** a `SHAPResult`, **When** `plot_shap_bar(r)` is called, **Then**
   the returned `Axes` ranks features by mean absolute SHAP.

### User Story 2 — Hierarchical reporting figures (Priority: P2)

A user wants per-level views of a `HierarchicalResult` for a written
report.

**Independent Test**: each of `plot_hierarchy_bar`, `plot_hierarchy_heatmap`,
`plot_hierarchy_violin`, `plot_hierarchy_summary`, `plot_hierarchy_tree`,
`plot_hierarchy_comparison` can be called on a small synthetic
`HierarchicalResult` without raising.

## Functional Requirements

### FR-007.1 Standard plots (`xeries.visualization.plots`)

| Function | Input | Returns |
| --- | --- | --- |
| `plot_importance_bar(result, top_n=None, ax=None)` | `FeatureImportanceResult` | `matplotlib.axes.Axes` |
| `plot_importance_comparison(*results, labels=None, ax=None)` | several `FeatureImportanceResult`s | `matplotlib.axes.Axes` |
| `plot_importance_heatmap(per_series_df, ax=None)` | DataFrame with series × feature | `matplotlib.axes.Axes` |
| `plot_importance_per_series(per_series_df, ax=None)` | DataFrame with series × feature | `matplotlib.axes.Axes` |
| `plot_shap_bar(shap_result, top_n=None, ax=None)` | `SHAPResult` | `matplotlib.axes.Axes` |
| `plot_shap_summary(shap_result, ax=None)` | `SHAPResult` | `matplotlib.axes.Axes` |

### FR-007.2 Hierarchical plots (`xeries.visualization.hierarchy_plots`)

| Function | Input | Returns |
| --- | --- | --- |
| `plot_hierarchy_bar(result, level, ax=None)` | `HierarchicalResult` | `Axes` |
| `plot_hierarchy_comparison(*results, level, labels=None, ax=None)` | several | `Axes` |
| `plot_hierarchy_heatmap(result, level, ax=None)` | `HierarchicalResult` | `Axes` |
| `plot_hierarchy_summary(result, ax=None)` | `HierarchicalResult` | `Axes` |
| `plot_hierarchy_tree(hierarchy_def, ax=None)` | `HierarchyDefinition` | `Axes` |
| `plot_hierarchy_violin(result, level, ax=None)` | `HierarchicalResult` | `Axes` |

### FR-007.3 Cross-cutting rules

- All plotting helpers MUST accept an optional `ax` argument and return the
  axes they drew on.
- They MUST NOT call `plt.show()`. Display is the caller's responsibility.
- They MUST NOT mutate the input result objects.
- They MUST NOT depend on any optional plotting backend other than
  `matplotlib`. (`seaborn` may be used internally; if so it MUST be a hard
  runtime dep, not optional.)

## Public API surface

```python
from xeries.visualization import (
    plot_importance_bar,
    plot_importance_comparison,
    plot_importance_heatmap,
    plot_importance_per_series,
    plot_shap_bar,
    plot_shap_summary,
    plot_hierarchy_bar,
    plot_hierarchy_comparison,
    plot_hierarchy_heatmap,
    plot_hierarchy_summary,
    plot_hierarchy_tree,
    plot_hierarchy_violin,
)
# All re-exported at top level on `xeries`.
```

Source files: `src/xeries/visualization/plots.py`,
`src/xeries/visualization/hierarchy_plots.py`.

## Test coverage

- `tests/visualization/test_hierarchy_plots.py` — smoke tests that each
  hierarchy plot draws on a synthetic result without raising.
- The standard plots are exercised through the example notebooks under
  `examples/`. Adding direct smoke tests for the standard plots is a
  reasonable follow-up but not blocked.

## Out of scope / future work

- Interactive plotting (Plotly / Altair / Bokeh) is intentionally out of
  scope. Users who need interactivity should pull `result.to_dataframe()` and
  feed their backend of choice.
- The dashboard application alluded to in the optional `jinja2` dep is
  **not** part of this spec — it would be its own feature with its own spec
  if/when it lands.

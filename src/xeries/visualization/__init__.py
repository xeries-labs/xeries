"""Visualization utilities for feature importance."""

from xeries.visualization.hierarchy_plots import (
    plot_hierarchy_bar,
    plot_hierarchy_comparison,
    plot_hierarchy_heatmap,
    plot_hierarchy_summary,
    plot_hierarchy_tree,
    plot_hierarchy_violin,
)
from xeries.visualization.plots import (
    plot_importance_bar,
    plot_importance_comparison,
    plot_importance_heatmap,
    plot_importance_per_series,
    plot_shap_bar,
    plot_shap_summary,
)

__all__ = [
    # Hierarchical plots
    "plot_hierarchy_bar",
    "plot_hierarchy_comparison",
    "plot_hierarchy_heatmap",
    "plot_hierarchy_summary",
    "plot_hierarchy_tree",
    "plot_hierarchy_violin",
    # Standard plots
    "plot_importance_bar",
    "plot_importance_comparison",
    "plot_importance_heatmap",
    "plot_importance_per_series",
    "plot_shap_bar",
    "plot_shap_summary",
]

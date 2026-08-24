"""Hierarchical feature importance for multi-level time series data.

This module provides tools for computing and aggregating feature importance
across hierarchical structures commonly found in demand forecasting and
other multi-series time series problems.

Key components:
    - HierarchyDefinition: Define the hierarchical structure of your data
    - HierarchicalAggregator: Aggregate importance values across hierarchy levels
    - HierarchicalExplainer: Unified wrapper combining base explainers with hierarchy
    - HierarchicalResult: Container for hierarchical importance results

Example:
    >>> from xeries.hierarchy import HierarchyDefinition, HierarchicalExplainer
    >>> from xeries.importance import ConditionalSHAP
    >>>
    >>> # Define hierarchy (M5 dataset style)
    >>> hierarchy = HierarchyDefinition(
    ...     levels=["state", "store", "product"],
    ...     columns=["state_id", "store_id", "item_id"]
    ... )
    >>>
    >>> # Create hierarchical explainer
    >>> base_explainer = ConditionalSHAP(model, X_train, series_col='level')
    >>> explainer = HierarchicalExplainer(base_explainer, hierarchy)
    >>>
    >>> # Compute hierarchical explanations
    >>> result = explainer.explain(X_test)
    >>>
    >>> # Access results at different levels
    >>> print(result.get_global())
    >>> print(result.get_level_df("state"))
"""

from xeries.hierarchy.aggregator import HierarchicalAggregator
from xeries.hierarchy.definition import HierarchyDefinition
from xeries.hierarchy.explainer import HierarchicalExplainer
from xeries.hierarchy.types import HierarchicalResult

__all__ = [
    "HierarchicalAggregator",
    "HierarchicalExplainer",
    "HierarchicalResult",
    "HierarchyDefinition",
]

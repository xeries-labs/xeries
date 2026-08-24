"""Hierarchical explainer wrapper for multi-level importance analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from xeries.core.base import BaseExplainer
from xeries.core.types import FeatureImportanceResult, SHAPResult
from xeries.hierarchy.aggregator import HierarchicalAggregator
from xeries.hierarchy.definition import HierarchyDefinition
from xeries.hierarchy.types import HierarchicalResult

if TYPE_CHECKING:
    pass


class HierarchicalExplainer:
    """Wrapper that adds hierarchical aggregation to any base explainer.

    Provides a unified workflow for computing feature importance and
    aggregating results across hierarchy levels:

    1. Define hierarchy structure
    2. Compute base explanations using wrapped explainer
    3. Aggregate across hierarchy levels
    4. Access results at any level

    This explainer is designed for multi-series forecasting scenarios
    where data has a natural hierarchical structure (e.g., region -> store -> product).

    Example:
        >>> from xeries.hierarchy import HierarchyDefinition, HierarchicalExplainer
        >>> from xeries.importance import ConditionalSHAP
        >>>
        >>> hierarchy = HierarchyDefinition(
        ...     levels=["state", "store", "product"],
        ...     columns=["state_id", "store_id", "item_id"]
        ... )
        >>> base_explainer = ConditionalSHAP(model, X, series_col='level')
        >>> explainer = HierarchicalExplainer(base_explainer, hierarchy)
        >>> result = explainer.explain(X_test)
        >>>
        >>> # Access results at different levels
        >>> print(result.get_global())
        >>> print(result.get_level_df("state"))

    Attributes:
        base_explainer: The underlying explainer (e.g. ``ConditionalSHAP``
            or ``ConditionalPermutationImportance``).
        hierarchy: HierarchyDefinition specifying the hierarchical structure.
    """

    def __init__(
        self,
        base_explainer: BaseExplainer,
        hierarchy: HierarchyDefinition,
    ) -> None:
        """Initialize the hierarchical explainer.

        Args:
            base_explainer: A base explainer instance (ConditionalSHAP or
                ConditionalPermutationImportance).
            hierarchy: HierarchyDefinition specifying the hierarchical structure.
        """
        self.base_explainer = base_explainer
        self.hierarchy = hierarchy
        self._aggregator = HierarchicalAggregator(hierarchy)

        self._explainer_type = self._detect_explainer_type()

    def _detect_explainer_type(self) -> str:
        """Detect the type of base explainer."""
        class_name = self.base_explainer.__class__.__name__

        if "SHAP" in class_name or "Shap" in class_name:
            return "shap"
        elif "Permutation" in class_name:
            return "permutation"
        else:
            return "unknown"

    def explain(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
        levels: list[str] | None = None,
        include_raw: bool = True,
        **kwargs: Any,
    ) -> HierarchicalResult:
        """Compute explanations and aggregate across hierarchy levels.

        1. Calls base_explainer.explain() to get raw results
        2. Aggregates using HierarchicalAggregator
        3. Returns HierarchicalResult

        Args:
            X: Input features DataFrame to explain.
            y: Target values (required for permutation importance).
            levels: Which hierarchy levels to include in aggregation.
                If None, includes 'global' plus all levels from hierarchy definition.
            include_raw: Whether to store raw values for distribution plots
                (only applicable for SHAP results).
            **kwargs: Additional arguments passed to base explainer.

        Returns:
            HierarchicalResult containing aggregated importance at each level.

        Raises:
            ValueError: If y is required but not provided.
        """
        self.hierarchy.validate_data(X)

        if self._explainer_type == "shap":
            base_result = self.base_explainer.explain(X, **kwargs)
            if not isinstance(base_result, SHAPResult):
                raise TypeError(f"Expected SHAPResult from base explainer, got {type(base_result)}")
            return self._aggregator.aggregate_shap(
                base_result, X, levels=levels, include_raw=include_raw
            )

        elif self._explainer_type == "permutation":
            if y is None:
                raise ValueError("y (target values) must be provided for permutation importance")
            base_result = self.base_explainer.explain(X, y, **kwargs)
            if not isinstance(base_result, FeatureImportanceResult):
                raise TypeError(
                    f"Expected FeatureImportanceResult from base explainer, got {type(base_result)}"
                )
            return self._aggregator.aggregate_importance(
                base_result,
                X,
                y,
                levels=levels,
                model=getattr(self.base_explainer, "model", None),
                metric=getattr(self.base_explainer, "metric", "mse"),
                n_repeats=getattr(self.base_explainer, "n_repeats", 5),
            )

        else:
            base_result = self.base_explainer.explain(X, y, **kwargs)
            if isinstance(base_result, SHAPResult):
                return self._aggregator.aggregate_shap(
                    base_result, X, levels=levels, include_raw=include_raw
                )
            elif isinstance(base_result, FeatureImportanceResult):
                if y is None:
                    raise ValueError("y must be provided for importance aggregation")
                return self._aggregator.aggregate_importance(base_result, X, y, levels=levels)
            else:
                raise TypeError(
                    f"Base explainer returned unsupported result type: {type(base_result)}"
                )

    def explain_level(
        self,
        X: pd.DataFrame,
        level: str,
        cohort: str | None = None,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Get importance for a specific hierarchy level (convenience method).

        Args:
            X: Input features DataFrame.
            level: The hierarchy level to get results for.
            cohort: Specific cohort at that level. If None, returns all cohorts.
            y: Target values (required for permutation importance).
            **kwargs: Additional arguments passed to explain().

        Returns:
            DataFrame with features as columns.
            If cohort is specified, returns single-row DataFrame.
            Otherwise, returns DataFrame with cohorts as rows.
        """
        result = self.explain(X, y=y, levels=[level], **kwargs)

        df = result.get_level_df(level)

        if cohort is not None:
            if cohort not in df.index:
                raise KeyError(
                    f"Cohort '{cohort}' not found at level '{level}'. Available: {list(df.index)}"
                )
            return df.loc[[cohort]]

        return df

    def explain_per_cohort(
        self,
        X: pd.DataFrame,
        level: str,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, HierarchicalResult]:
        """Compute separate explanations for each cohort at a level.

        Useful when you want detailed analysis within each cohort,
        including sub-levels below the specified level.

        Args:
            X: Input features DataFrame.
            level: The hierarchy level to split by.
            y: Target values (required for permutation importance).
            **kwargs: Additional arguments passed to explain().

        Returns:
            Dictionary mapping cohort names to HierarchicalResult objects.
        """
        cohorts = self.hierarchy.get_cohorts(X, level)
        level_idx = self.hierarchy.levels.index(level)
        sub_levels = self.hierarchy.levels[level_idx + 1 :]

        results = {}
        for cohort_name, indices in cohorts.items():
            cohort_X = X.loc[indices]
            cohort_y = (
                y.loc[indices]
                if isinstance(y, pd.Series)
                else (y[X.index.isin(indices)] if y is not None else None)
            )

            result = self.explain(
                cohort_X,
                y=cohort_y,
                levels=["global", *sub_levels] if sub_levels else ["global"],
                **kwargs,
            )
            results[cohort_name] = result

        return results

    def get_aggregator(self) -> HierarchicalAggregator:
        """Get the underlying aggregator for advanced usage.

        Returns:
            The HierarchicalAggregator instance.
        """
        return self._aggregator

    def compare_cohorts(
        self,
        X: pd.DataFrame,
        level: str,
        y: pd.Series | np.ndarray | None = None,
        top_n: int = 10,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compare feature importance across cohorts at a given level.

        Args:
            X: Input features DataFrame.
            level: The hierarchy level to compare.
            y: Target values (required for permutation importance).
            top_n: Number of top features to include.
            **kwargs: Additional arguments passed to explain().

        Returns:
            DataFrame with features as rows, cohorts as columns,
            containing importance values.
        """
        result = self.explain(X, y=y, levels=[level], **kwargs)
        df = result.get_level_df(level).T

        mean_importance = df.mean(axis=1)
        top_features = mean_importance.nlargest(top_n).index.tolist()

        return df.loc[top_features]

    def feature_ranking_stability(
        self,
        X: pd.DataFrame,
        level: str,
        y: pd.Series | np.ndarray | None = None,
        top_n: int = 10,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Analyze how feature rankings vary across cohorts.

        Args:
            X: Input features DataFrame.
            level: The hierarchy level to analyze.
            y: Target values (required for permutation importance).
            top_n: Number of top features to consider for ranking.
            **kwargs: Additional arguments passed to explain().

        Returns:
            DataFrame with features as rows and columns:
            - mean_rank: Average rank across cohorts
            - std_rank: Standard deviation of rank
            - min_rank: Best (lowest) rank achieved
            - max_rank: Worst (highest) rank achieved
        """
        result = self.explain(X, y=y, levels=[level], **kwargs)
        level_df = result.get_level_df(level)

        rankings = pd.DataFrame(index=result.features)

        for cohort in level_df.index:
            cohort_importance = level_df.loc[cohort]
            cohort_ranks = cohort_importance.rank(ascending=False)
            rankings[cohort] = cohort_ranks

        stability = pd.DataFrame(
            {
                "mean_rank": rankings.mean(axis=1),
                "std_rank": rankings.std(axis=1),
                "min_rank": rankings.min(axis=1),
                "max_rank": rankings.max(axis=1),
            }
        )

        stability = stability.sort_values("mean_rank")

        if top_n is not None:
            stability = stability.head(top_n)

        return stability

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HierarchicalExplainer("
            f"base={self.base_explainer.__class__.__name__}, "
            f"hierarchy_levels={self.hierarchy.levels})"
        )

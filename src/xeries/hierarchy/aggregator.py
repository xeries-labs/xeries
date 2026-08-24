"""Hierarchical aggregation of feature importance results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from xeries.hierarchy.definition import HierarchyDefinition
from xeries.hierarchy.types import HierarchicalResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from xeries.core.types import FeatureImportanceResult, SHAPResult


class HierarchicalAggregator:
    """Aggregates feature importance across hierarchy levels.

    Implements cohort-based aggregation using the formula from the article:

        phi_i(C_k) = (1/|C_k|) * sum(|phi_i(x)|) for x in C_k

    where:
        - phi_i(C_k) is the importance of feature i for cohort C_k
        - |C_k| is the number of samples in cohort C_k
        - |phi_i(x)| is the absolute importance of feature i for sample x

    This aggregator can be used standalone with pre-computed results
    or integrated into HierarchicalExplainer for a unified workflow.

    Example:
        >>> hierarchy = HierarchyDefinition(
        ...     levels=["state", "store"],
        ...     columns=["state_id", "store_id"]
        ... )
        >>> aggregator = HierarchicalAggregator(hierarchy)
        >>> result = aggregator.aggregate_shap(shap_result, data)
    """

    def __init__(self, hierarchy: HierarchyDefinition) -> None:
        """Initialize the aggregator.

        Args:
            hierarchy: HierarchyDefinition specifying the hierarchical structure.
        """
        self.hierarchy = hierarchy

    def aggregate_shap(
        self,
        shap_result: SHAPResult,
        data: pd.DataFrame,
        levels: list[str] | None = None,
        include_raw: bool = True,
    ) -> HierarchicalResult:
        """Aggregate SHAP values at each hierarchy level.

        For each level and cohort, computes mean absolute SHAP values
        and optionally stores raw values for distribution plots.

        Args:
            shap_result: SHAPResult from ConditionalSHAP or similar.
            data: DataFrame with hierarchy information (must have same index as explained data).
            levels: Which hierarchy levels to include. If None, includes all levels plus 'global'.
            include_raw: Whether to include raw SHAP values for distribution plots.

        Returns:
            HierarchicalResult with aggregated importance at each level.
        """
        if levels is None:
            levels = ["global", *self.hierarchy.levels]
        elif "global" not in levels:
            levels = ["global", *levels]

        feature_names = shap_result.feature_names
        shap_values = shap_result.shap_values

        if len(data) != len(shap_values):
            raise ValueError(
                f"Data length ({len(data)}) must match SHAP values length ({len(shap_values)})"
            )

        importance_by_level: dict[str, dict[str, dict[str, float]]] = {}
        raw_values_by_level: dict[str, dict[str, NDArray[np.floating[Any]]]] = {}
        feature_values_by_level: dict[str, dict[str, NDArray[np.floating[Any]]]] = {}
        cohort_sizes: dict[str, dict[str, int]] = {}

        for level in levels:
            importance_by_level[level] = {}
            raw_values_by_level[level] = {}
            feature_values_by_level[level] = {}
            cohort_sizes[level] = {}

            if level == "global":
                cohorts = {"all": data.index}
            else:
                cohorts = self.hierarchy.get_cohorts(data, level)

            for cohort_name, indices in cohorts.items():
                mask = data.index.isin(indices)
                cohort_shap = shap_values[mask]
                cohort_sizes[level][cohort_name] = len(cohort_shap)

                mean_abs_shap = np.abs(cohort_shap).mean(axis=0)
                importance_by_level[level][cohort_name] = dict(
                    zip(feature_names, mean_abs_shap, strict=True)
                )

                if include_raw:
                    raw_values_by_level[level][cohort_name] = cohort_shap
                    cohort_data = (
                        data.loc[indices, feature_names].values
                        if all(f in data.columns for f in feature_names)
                        else None
                    )
                    if cohort_data is not None:
                        feature_values_by_level[level][cohort_name] = cohort_data

        return HierarchicalResult(
            levels=levels,
            features=feature_names,
            importance_by_level=importance_by_level,
            raw_values_by_level=raw_values_by_level if include_raw else None,
            feature_values_by_level=(
                feature_values_by_level if include_raw and feature_values_by_level else None
            ),
            method="shap",
            _cohort_sizes=cohort_sizes,
        )

    def aggregate_importance(
        self,
        importance_result: FeatureImportanceResult,
        data: pd.DataFrame,
        y: pd.Series | np.ndarray,
        levels: list[str] | None = None,
        model: Any = None,
        metric: str = "mse",
        n_repeats: int = 5,
    ) -> HierarchicalResult:
        """Aggregate permutation importance at each hierarchy level.

        For permutation importance, we need to re-compute importance
        for each cohort separately to get meaningful aggregated values.

        Args:
            importance_result: FeatureImportanceResult with global importance.
            data: DataFrame with hierarchy information.
            y: Target values corresponding to data.
            levels: Which hierarchy levels to include.
            model: Model to use for re-computing cohort-level importance.
                   If None, uses mean importance from global result.
            metric: Metric for permutation importance ('mse', 'mae', 'rmse', 'r2').
            n_repeats: Number of permutation repeats.

        Returns:
            HierarchicalResult with aggregated importance at each level.
        """
        if levels is None:
            levels = ["global", *self.hierarchy.levels]
        elif "global" not in levels:
            levels = ["global", *levels]

        feature_names = importance_result.feature_names
        importances = importance_result.importances

        importance_by_level: dict[str, dict[str, dict[str, float]]] = {}
        cohort_sizes: dict[str, dict[str, int]] = {}

        importance_by_level["global"] = {"all": dict(zip(feature_names, importances, strict=True))}
        cohort_sizes["global"] = {"all": len(data)}

        for level in levels:
            if level == "global":
                continue

            importance_by_level[level] = {}
            cohort_sizes[level] = {}

            cohorts = self.hierarchy.get_cohorts(data, level)

            for cohort_name, indices in cohorts.items():
                cohort_data = data.loc[indices]
                cohort_y = (
                    y.loc[indices] if isinstance(y, pd.Series) else y[data.index.isin(indices)]
                )
                cohort_sizes[level][cohort_name] = len(cohort_data)

                if model is not None:
                    cohort_importance = self._compute_cohort_importance(
                        model, cohort_data, cohort_y, feature_names, metric, n_repeats
                    )
                else:
                    cohort_importance = dict(zip(feature_names, importances, strict=True))

                importance_by_level[level][cohort_name] = cohort_importance

        return HierarchicalResult(
            levels=levels,
            features=feature_names,
            importance_by_level=importance_by_level,
            raw_values_by_level=None,
            feature_values_by_level=None,
            method="permutation",
            _cohort_sizes=cohort_sizes,
        )

    def _compute_cohort_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        feature_names: list[str],
        metric: str,
        n_repeats: int,
    ) -> dict[str, float]:
        """Compute permutation importance for a single cohort."""
        from xeries.importance import ConditionalPermutationImportance

        explainer = ConditionalPermutationImportance(
            model=model,
            metric=metric,
            n_repeats=n_repeats,
        )

        result = explainer.explain(X[feature_names], y)
        return dict(zip(result.feature_names, result.importances, strict=True))

    def aggregate_from_per_series(
        self,
        per_series_results: dict[str, FeatureImportanceResult],
        data: pd.DataFrame,
        levels: list[str] | None = None,
    ) -> HierarchicalResult:
        """Aggregate pre-computed per-series importance to hierarchy levels.

        Useful when you have already computed importance per series and want
        to aggregate up the hierarchy without re-computing.

        Args:
            per_series_results: Dict mapping series_id to FeatureImportanceResult.
            data: DataFrame with hierarchy information.
            levels: Which hierarchy levels to include.

        Returns:
            HierarchicalResult with aggregated importance.
        """
        if levels is None:
            levels = ["global", *self.hierarchy.levels]
        elif "global" not in levels:
            levels = ["global", *levels]

        if not per_series_results:
            raise ValueError("per_series_results cannot be empty")

        first_result = next(iter(per_series_results.values()))
        feature_names = first_result.feature_names

        all_importances: dict[str, NDArray[np.floating[Any]]] = {}
        for series_id, result in per_series_results.items():
            all_importances[series_id] = result.importances

        importance_by_level: dict[str, dict[str, dict[str, float]]] = {}
        cohort_sizes: dict[str, dict[str, int]] = {}

        importance_by_level["global"] = {}
        cohort_sizes["global"] = {}

        all_imps = np.array(list(all_importances.values()))
        global_mean = all_imps.mean(axis=0)
        importance_by_level["global"]["all"] = dict(zip(feature_names, global_mean, strict=True))
        cohort_sizes["global"]["all"] = len(per_series_results)

        for level in levels:
            if level == "global":
                continue

            importance_by_level[level] = {}
            cohort_sizes[level] = {}

            cohorts = self.hierarchy.get_cohorts(data, level)

            series_ids = self.hierarchy._get_series_ids(data)
            series_to_cohort: dict[str, str] = {}
            for cohort_name, indices in cohorts.items():
                cohort_series = series_ids.loc[indices].unique()
                for sid in cohort_series:
                    series_to_cohort[str(sid)] = cohort_name

            cohort_importances: dict[str, list[NDArray[np.floating[Any]]]] = {
                name: [] for name in cohorts
            }

            for series_id, imps in all_importances.items():
                if series_id in series_to_cohort:
                    cohort_name = series_to_cohort[series_id]
                    cohort_importances[cohort_name].append(imps)

            for cohort_name, imp_list in cohort_importances.items():
                if imp_list:
                    cohort_mean = np.array(imp_list).mean(axis=0)
                    importance_by_level[level][cohort_name] = dict(
                        zip(feature_names, cohort_mean, strict=True)
                    )
                    cohort_sizes[level][cohort_name] = len(imp_list)

        return HierarchicalResult(
            levels=levels,
            features=feature_names,
            importance_by_level=importance_by_level,
            raw_values_by_level=None,
            feature_values_by_level=None,
            method="aggregated",
            _cohort_sizes=cohort_sizes,
        )

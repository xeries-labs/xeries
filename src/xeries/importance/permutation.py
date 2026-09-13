"""Conditional Permutation Feature Importance implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from xeries.core.base import BasePartitioner, MetricBasedExplainer
from xeries.core.types import (
    ArrayLike,
    FeatureImportanceResult,
    GroupLabels,
    MetricFunction,
    ModelProtocol,
)
from xeries.partitioners.tree import TreePartitioner

if TYPE_CHECKING:
    pass


class ConditionalPermutationImportance(MetricBasedExplainer):
    """Conditional Permutation Feature Importance calculator.

    This implements conditional permutation importance where feature values
    are only shuffled within defined subgroups, preserving the correlation
    structure between features.

    Supports two strategies:
    - 'auto': Uses tree-based cs-PFI to automatically learn subgroups
    - 'manual': Uses pre-defined groups provided by the user

    Example:
        >>> explainer = ConditionalPermutationImportance(model, metric='mse')
        >>> result = explainer.explain(X, y, features=['lag_1', 'lag_2'])
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        model: ModelProtocol,
        metric: MetricFunction | str = "mse",
        strategy: str = "auto",
        partitioner: BasePartitioner | None = None,
        n_repeats: int = 5,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> None:
        """Initialize the conditional permutation importance calculator.

        Args:
            model: A model with a predict method.
            metric: Scoring metric ('mse', 'mae', 'rmse', 'r2') or callable.
            strategy: Grouping strategy ('auto' for tree-based, 'manual' for user-defined).
            partitioner: Custom partitioner instance. If None, uses TreePartitioner for 'auto'.
            n_repeats: Number of times to repeat permutation for each feature.
            n_jobs: Number of parallel jobs (-1 for all cores).
            random_state: Random seed for reproducibility.
        """
        super().__init__(model, metric, random_state)
        self.strategy = strategy
        self.partitioner = partitioner
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs

    def explain(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        features: list[str] | None = None,
        groups: GroupLabels | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> FeatureImportanceResult:  # type: ignore[override]
        """Compute conditional permutation importance for features.

        Args:
            X: Input features DataFrame.
            y: Target values.
            features: List of features to compute importance for.
                If None, uses all columns in X.
            groups: Pre-defined group labels for 'manual' strategy.
                Required when strategy='manual' and no partitioner is provided.

        Returns:
            FeatureImportanceResult containing importance scores.
        """
        y_array = np.asarray(y)
        features = features or list(X.columns)

        baseline_pred = self.model.predict(X)
        baseline_score = self.metric(y_array, baseline_pred)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._compute_feature_importance)(X, y_array, feature, baseline_score, groups)
            for feature in features
        )

        importances = []
        stds = []
        permuted_scores: dict[str, list[float]] = {}

        for feature, scores in zip(features, results, strict=True):
            importance_values = [score - baseline_score for score in scores]
            importances.append(np.mean(importance_values))
            stds.append(np.std(importance_values))
            permuted_scores[feature] = scores

        return FeatureImportanceResult(
            feature_names=features,
            importances=np.array(importances),
            std=np.array(stds),
            baseline_score=baseline_score,
            permuted_scores=permuted_scores,
            method="conditional_permutation",
            n_repeats=self.n_repeats,
        )

    def _compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: NDArray[np.floating[Any]],
        feature: str,
        baseline_score: float,
        groups: GroupLabels | None,
    ) -> list[float]:
        """Compute importance for a single feature across repeats."""
        group_labels = self._get_groups(X, feature, groups)

        scores = []
        for rng in self._repeat_generators():
            X_permuted = self._conditional_permute(X, feature, group_labels, rng)
            permuted_pred = self.model.predict(X_permuted)
            score = self.metric(y, permuted_pred)
            scores.append(score)

        return scores

    def _repeat_generators(self):
        """Yield one independent RNG per permutation repeat.

        Uses :class:`numpy.random.SeedSequence` to derive ``n_repeats``
        statistically-independent child streams from a single seed, mirroring
        the approach used by scikit-learn's permutation importance. This keeps
        every repeat distinct while remaining fully reproducible for any integer
        ``random_state`` (including ``0``, which is falsy and would be mishandled
        by a truthiness check). When ``random_state`` is ``None`` the streams are
        non-deterministic, as expected.

        Yields:
            np.random.Generator: A fresh generator for each repeat.
        """
        seed_sequence = np.random.SeedSequence(self.random_state)
        for child in seed_sequence.spawn(self.n_repeats):
            yield np.random.default_rng(child)

    def _get_groups(
        self,
        X: pd.DataFrame,
        feature: str,
        groups: GroupLabels | None,
    ) -> NDArray[np.intp]:
        """Get group labels for conditional permutation."""
        if groups is not None:
            return np.asarray(groups).astype(np.intp)

        if self.partitioner is not None:
            return self.partitioner.fit_get_groups(X, feature)

        if self.strategy == "auto":
            partitioner = TreePartitioner(random_state=self.random_state)
            return partitioner.fit_get_groups(X, feature)

        raise ValueError("For strategy='manual', either provide 'groups' or a 'partitioner'")

    def explain_per_series(
        self,
        X: pd.DataFrame,
        y: ArrayLike,
        series_col: str,
        features: list[str] | None = None,
        min_samples: int = 10,
    ) -> dict[Any, FeatureImportanceResult]:
        """Compute conditional permutation importance separately for each series.

        This method filters the data by each unique series ID and computes
        feature importance independently for each series. Permutation is
        performed only within each individual series.

        Args:
            X: Input features DataFrame.
            y: Target values.
            series_col: Name of the column or MultiIndex level containing series IDs.
            features: List of features to compute importance for.
                If None, uses all columns except series_col.
            min_samples: Minimum number of samples required per series.
                Series with fewer samples are skipped.

        Returns:
            Dictionary mapping series IDs to FeatureImportanceResult objects.

        Example:
            >>> explainer = ConditionalPermutationImportance(model, metric='mse')
            >>> results = explainer.explain_per_series(X, y, series_col='level')
            >>> for series_id, result in results.items():
            ...     print(f"{series_id}: {result.to_dataframe()}")
        """
        y_array = np.asarray(y)

        series_ids = self._get_series_ids_from_data(X, series_col)
        unique_series = series_ids.unique()

        if features is None:
            exclude_cols = {series_col}
            features = [c for c in X.columns if c not in exclude_cols]

        results: dict[Any, FeatureImportanceResult] = {}

        for series_id in unique_series:
            mask = series_ids == series_id
            X_series = X.loc[mask]
            y_series = y_array[mask]

            if len(X_series) < min_samples:
                continue

            result = self._compute_series_importance(X_series, y_series, features)
            results[series_id] = result

        return results

    def _get_series_ids_from_data(
        self,
        X: pd.DataFrame,
        series_col: str,
    ) -> pd.Series:
        """Extract series identifiers from DataFrame."""
        if isinstance(X.index, pd.MultiIndex) and series_col in X.index.names:
            return X.index.get_level_values(series_col).to_series(index=X.index)

        if series_col in X.columns:
            return X[series_col]

        raise KeyError(f"Series column '{series_col}' not found in DataFrame columns or index")

    def _compute_series_importance(
        self,
        X: pd.DataFrame,
        y: NDArray[np.floating[Any]],
        features: list[str],
    ) -> FeatureImportanceResult:
        """Compute importance for a single series (no conditional groups)."""
        baseline_pred = self.model.predict(X)
        baseline_score = self.metric(y, baseline_pred)

        importances = []
        stds = []
        permuted_scores: dict[str, list[float]] = {}

        for feature in features:
            if feature not in X.columns:
                continue

            scores = []
            for rng in self._repeat_generators():
                X_permuted = X.copy()
                X_permuted[feature] = rng.permutation(X[feature].to_numpy())
                permuted_pred = self.model.predict(X_permuted)
                score = self.metric(y, permuted_pred)
                scores.append(score)

            importance_values = [score - baseline_score for score in scores]
            importances.append(np.mean(importance_values))
            stds.append(np.std(importance_values))
            permuted_scores[feature] = scores

        valid_features = [f for f in features if f in X.columns]

        return FeatureImportanceResult(
            feature_names=valid_features,
            importances=np.array(importances),
            std=np.array(stds),
            baseline_score=baseline_score,
            permuted_scores=permuted_scores,
            method="per_series_permutation",
            n_repeats=self.n_repeats,
        )

    def _conditional_permute(
        self,
        X: pd.DataFrame,
        feature: str,
        groups: NDArray[np.intp],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Permute feature values within groups.

        Args:
            X: Input DataFrame.
            feature: Feature column to permute.
            groups: Group labels for each row.
            rng: Random number generator.

        Returns:
            DataFrame with permuted feature values.
        """
        X_permuted = X.copy()
        feature_values = X_permuted[feature].to_numpy()

        unique_groups = np.unique(groups)
        permuted_values = feature_values.copy()

        for group in unique_groups:
            mask = groups == group
            group_values = feature_values[mask]
            permuted_values[mask] = rng.permutation(group_values)

        X_permuted[feature] = permuted_values
        return X_permuted

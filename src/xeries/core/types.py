"""Type definitions for xeries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

ArrayLike = np.ndarray | pd.Series | pd.DataFrame
GroupLabels = np.ndarray | pd.Series | list[Any]


class ModelProtocol(Protocol):
    """Protocol for models that can be used with xeries explainers."""

    def predict(
        self, X: ArrayLike | pd.DataFrame
    ) -> NDArray[np.floating[Any]] | np.ndarray | pd.Series:
        """Make predictions on input data."""
        ...


@dataclass
class BaseResult:
    """Base class for all explainability results."""

    pass


@dataclass
class FeatureImportanceResult(BaseResult):
    """Container for feature importance results.

    Attributes:
        feature_names: List of feature names.
        importances: Array of importance scores for each feature.
        std: Standard deviations of importance scores (from multiple permutations).
        baseline_score: The baseline model score before permutation.
        permuted_scores: Dictionary mapping feature names to their permuted scores.
        method: The method used to compute importance ('permutation', 'shap', etc.).
        n_repeats: Number of permutation repeats used.
    """

    feature_names: list[str]
    importances: NDArray[np.floating[Any]]
    std: NDArray[np.floating[Any]] | None = None
    baseline_score: float = 0.0
    permuted_scores: dict[str, list[float]] = field(default_factory=dict)
    method: str = "permutation"
    n_repeats: int = 1

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data = {
            "feature": self.feature_names,
            "importance": self.importances,
        }
        if self.std is not None:
            data["std"] = self.std
        return pd.DataFrame(data).sort_values("importance", ascending=False)


@dataclass
class SHAPResult(BaseResult):
    """Container for SHAP explanation results.

    Attributes:
        shap_values: SHAP values array with shape (n_samples, n_features).
        base_values: Base/expected values for each sample.
        feature_names: List of feature names.
        data: The input data used for explanation.
        series_ids: Optional series identifiers for each sample (for hierarchical aggregation).
    """

    shap_values: NDArray[np.floating[Any]]
    base_values: NDArray[np.floating[Any]]
    feature_names: list[str]
    data: ArrayLike
    series_ids: pd.Series | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert SHAP values to a pandas DataFrame.

        Returns:
            DataFrame with samples as rows, features as columns, values are SHAP values.
        """
        return pd.DataFrame(
            self.shap_values,
            columns=self.feature_names,
        )

    def mean_abs_shap(self) -> pd.DataFrame:
        """Compute mean absolute SHAP values per feature."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)

    def mean_abs_shap_by_series(self) -> pd.DataFrame:
        """Compute mean absolute SHAP values per feature, grouped by series.

        Returns:
            DataFrame with series_id as index, features as columns, values are mean |SHAP|.

        Raises:
            ValueError: If series_ids is not set.
        """
        if self.series_ids is None:
            raise ValueError(
                "series_ids not set. Use ConditionalSHAP with series_col to track series."
            )

        result_data = {}
        for series_id in self.series_ids.unique():
            mask = self.series_ids == series_id
            series_shap = self.shap_values[mask]
            mean_abs = np.abs(series_shap).mean(axis=0)
            result_data[series_id] = dict(zip(self.feature_names, mean_abs, strict=True))

        return pd.DataFrame(result_data).T


@dataclass
class RefutationResult(BaseResult):
    """Container for causal refutation test results.

    Attributes:
        method: The refutation method used (e.g. 'placebo_treatment').
        original_effect: The original estimated causal effect.
        refuted_effect: The effect estimated under the refutation.
        p_value: Statistical significance of the refutation test.
        passed: True if the refutation confirms the original estimate.
    """

    method: str = ""
    original_effect: float = 0.0
    refuted_effect: float = 0.0
    p_value: float | None = None
    passed: bool = True


@dataclass
class CausalResult(BaseResult):
    """Container for causal feature importance results.

    Attributes:
        feature_names: List of treatment feature names.
        treatment_effects: Average Treatment Effect per feature.
        confidence_intervals: (n_features, 2) array of (lower, upper) bounds.
        p_values: Statistical significance per feature.
        causal_graph: The causal DAG used (networkx DiGraph or similar).
        estimator_name: Name of the causal estimator used.
        refutation: Optional refutation result for robustness check.
    """

    feature_names: list[str] = field(default_factory=list)
    treatment_effects: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([]))
    confidence_intervals: NDArray[np.floating[Any]] | None = None
    p_values: NDArray[np.floating[Any]] | None = None
    causal_graph: Any = None
    estimator_name: str = "causal_forest"
    refutation: RefutationResult | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data: dict[str, Any] = {
            "feature": self.feature_names,
            "treatment_effect": self.treatment_effects,
        }
        if self.confidence_intervals is not None:
            data["ci_lower"] = self.confidence_intervals[:, 0]
            data["ci_upper"] = self.confidence_intervals[:, 1]
        if self.p_values is not None:
            data["p_value"] = self.p_values
        return pd.DataFrame(data).sort_values("treatment_effect", ascending=False, key=abs)

    def significant_features(self, alpha: float = 0.05) -> list[str]:
        """Return features with statistically significant causal effects."""
        if self.p_values is None:
            return list(self.feature_names)
        return [
            name for name, p in zip(self.feature_names, self.p_values, strict=True) if p < alpha
        ]


# Type alias for metric functions that take true and predicted values and return a score.
MetricFunction: TypeAlias = Callable[[ArrayLike, ArrayLike], float | int]

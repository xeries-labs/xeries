"""Type definitions for hierarchical feature importance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class HierarchicalResult:
    """Container for hierarchical feature importance results.

    Stores aggregated importance values at multiple hierarchy levels,
    following the cohort-based mean absolute SHAP aggregation formula:

        phi_i(C_k) = (1/|C_k|) * sum(|phi_i(x)|) for x in C_k

    where C_k is a cohort at hierarchy level k.

    Attributes:
        levels: List of hierarchy level names (e.g., ['global', 'state', 'store']).
        features: List of feature names.
        importance_by_level: Nested dict mapping level -> cohort -> feature -> importance.
        raw_values_by_level: Raw SHAP/importance values for distribution plots.
            Maps level -> cohort -> array of shape (n_samples, n_features).
        feature_values_by_level: Feature values for each sample (for violin plots).
            Maps level -> cohort -> array of shape (n_samples, n_features).
        method: The explanation method used ('shap' or 'permutation').
    """

    levels: list[str]
    features: list[str]
    importance_by_level: dict[str, dict[str, dict[str, float]]]
    raw_values_by_level: dict[str, dict[str, NDArray[np.floating[Any]]]] | None = None
    feature_values_by_level: dict[str, dict[str, NDArray[np.floating[Any]]]] | None = None
    method: str = "shap"
    _cohort_sizes: dict[str, dict[str, int]] = field(default_factory=dict)

    def get_level_df(self, level: str) -> pd.DataFrame:
        """Get importance as DataFrame for a specific hierarchy level.

        Args:
            level: The hierarchy level name.

        Returns:
            DataFrame with cohorts as rows, features as columns,
            sorted by mean importance across features.

        Raises:
            KeyError: If level is not in the result.
        """
        if level not in self.importance_by_level:
            raise KeyError(f"Level '{level}' not found. Available: {self.levels}")

        level_data = self.importance_by_level[level]
        df = pd.DataFrame(level_data).T
        df = df[self.features]
        df["_mean_importance"] = df.mean(axis=1)
        df = df.sort_values("_mean_importance", ascending=False)
        return df.drop(columns=["_mean_importance"])

    def get_global(self) -> dict[str, float]:
        """Get global (all data) importance values.

        Returns:
            Dictionary mapping feature names to global importance scores.

        Raises:
            KeyError: If 'global' level is not present.
        """
        if "global" not in self.importance_by_level:
            raise KeyError("Global level not computed. Include 'global' in levels parameter.")

        return self.importance_by_level["global"]["all"]

    def get_global_df(self) -> pd.DataFrame:
        """Get global importance as a sorted DataFrame.

        Returns:
            DataFrame with columns ['feature', 'importance'], sorted by importance.
        """
        global_imp = self.get_global()
        df = pd.DataFrame(
            {"feature": list(global_imp.keys()), "importance": list(global_imp.values())}
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_cohorts_at_level(self, level: str) -> list[str]:
        """Get list of cohort names at a specific hierarchy level.

        Args:
            level: The hierarchy level name.

        Returns:
            List of cohort names.
        """
        if level not in self.importance_by_level:
            raise KeyError(f"Level '{level}' not found. Available: {self.levels}")

        return list(self.importance_by_level[level].keys())

    def get_feature_importance(self, level: str, cohort: str, feature: str) -> float:
        """Get importance value for a specific level/cohort/feature combination.

        Args:
            level: The hierarchy level name.
            cohort: The cohort name at that level.
            feature: The feature name.

        Returns:
            The importance value.
        """
        return self.importance_by_level[level][cohort][feature]

    def get_top_features(self, level: str, cohort: str, n: int = 10) -> list[tuple[str, float]]:
        """Get top N features by importance for a specific cohort.

        Args:
            level: The hierarchy level name.
            cohort: The cohort name.
            n: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples, sorted descending.
        """
        cohort_imp = self.importance_by_level[level][cohort]
        sorted_features = sorted(cohort_imp.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def get_raw_values(self, level: str, cohort: str) -> NDArray[np.floating[Any]] | None:
        """Get raw SHAP/importance values for a cohort (for distribution plots).

        Args:
            level: The hierarchy level name.
            cohort: The cohort name.

        Returns:
            Array of shape (n_samples, n_features) or None if not available.
        """
        if self.raw_values_by_level is None:
            return None
        if level not in self.raw_values_by_level:
            return None
        return self.raw_values_by_level[level].get(cohort)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert full results to a long-form DataFrame.

        Returns:
            DataFrame with columns ['level', 'cohort', 'feature', 'importance'].
        """
        rows = []
        for level in self.levels:
            for cohort, feature_imp in self.importance_by_level[level].items():
                for feature, importance in feature_imp.items():
                    rows.append(
                        {
                            "level": level,
                            "cohort": cohort,
                            "feature": feature,
                            "importance": importance,
                        }
                    )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """Return string representation."""
        n_levels = len(self.levels)
        n_features = len(self.features)
        total_cohorts = sum(len(cohorts) for cohorts in self.importance_by_level.values())
        return (
            f"HierarchicalResult(levels={n_levels}, features={n_features}, "
            f"total_cohorts={total_cohorts}, method='{self.method}')"
        )

"""Unit tests for HierarchicalResult."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xeries.hierarchy.types import HierarchicalResult


class TestHierarchicalResult:
    """Tests for HierarchicalResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> HierarchicalResult:
        """Create sample HierarchicalResult."""
        return HierarchicalResult(
            levels=["global", "state", "store"],
            features=["f1", "f2", "f3"],
            importance_by_level={
                "global": {
                    "all": {"f1": 0.5, "f2": 0.3, "f3": 0.2},
                },
                "state": {
                    "TX": {"f1": 0.4, "f2": 0.35, "f3": 0.25},
                    "WI": {"f1": 0.6, "f2": 0.25, "f3": 0.15},
                },
                "store": {
                    "TX_S1": {"f1": 0.35, "f2": 0.4, "f3": 0.25},
                    "TX_S2": {"f1": 0.45, "f2": 0.3, "f3": 0.25},
                    "WI_S1": {"f1": 0.55, "f2": 0.25, "f3": 0.2},
                    "WI_S2": {"f1": 0.65, "f2": 0.25, "f3": 0.1},
                },
            },
            raw_values_by_level={
                "global": {
                    "all": np.array(
                        [
                            [0.1, 0.2, 0.05],
                            [0.15, 0.25, 0.07],
                            [0.12, 0.22, 0.06],
                        ]
                    ),
                },
            },
            method="shap",
        )

    def test_get_global(self, sample_result: HierarchicalResult) -> None:
        """Test getting global importance."""
        global_imp = sample_result.get_global()

        assert global_imp == {"f1": 0.5, "f2": 0.3, "f3": 0.2}

    def test_get_global_missing_raises(self) -> None:
        """Test that missing global level raises KeyError."""
        result = HierarchicalResult(
            levels=["state"],
            features=["f1"],
            importance_by_level={"state": {"TX": {"f1": 0.5}}},
        )

        with pytest.raises(KeyError, match="Global level not computed"):
            result.get_global()

    def test_get_global_df(self, sample_result: HierarchicalResult) -> None:
        """Test getting global importance as DataFrame."""
        df = sample_result.get_global_df()

        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "importance" in df.columns
        assert df.iloc[0]["feature"] == "f1"
        assert df.iloc[0]["importance"] == 0.5

    def test_get_level_df(self, sample_result: HierarchicalResult) -> None:
        """Test getting level DataFrame."""
        df = sample_result.get_level_df("state")

        assert isinstance(df, pd.DataFrame)
        assert "TX" in df.index or "WI" in df.index
        assert "f1" in df.columns
        assert "f2" in df.columns
        assert "f3" in df.columns

    def test_get_level_df_invalid_raises(self, sample_result: HierarchicalResult) -> None:
        """Test that invalid level raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            sample_result.get_level_df("invalid")

    def test_get_cohorts_at_level(self, sample_result: HierarchicalResult) -> None:
        """Test getting cohort names at a level."""
        cohorts = sample_result.get_cohorts_at_level("store")

        assert "TX_S1" in cohorts
        assert "TX_S2" in cohorts
        assert "WI_S1" in cohorts
        assert "WI_S2" in cohorts

    def test_get_feature_importance(self, sample_result: HierarchicalResult) -> None:
        """Test getting specific feature importance."""
        imp = sample_result.get_feature_importance("state", "TX", "f1")

        assert imp == 0.4

    def test_get_top_features(self, sample_result: HierarchicalResult) -> None:
        """Test getting top features for a cohort."""
        top = sample_result.get_top_features("state", "TX", n=2)

        assert len(top) == 2
        assert top[0][0] == "f1"
        assert top[1][0] == "f2"

    def test_get_raw_values(self, sample_result: HierarchicalResult) -> None:
        """Test getting raw values."""
        raw = sample_result.get_raw_values("global", "all")

        assert raw is not None
        assert raw.shape == (3, 3)

    def test_get_raw_values_missing(self, sample_result: HierarchicalResult) -> None:
        """Test getting raw values when not available."""
        raw = sample_result.get_raw_values("state", "TX")

        assert raw is None

    def test_get_raw_values_no_data(self) -> None:
        """Test getting raw values when raw_values_by_level is None."""
        result = HierarchicalResult(
            levels=["global"],
            features=["f1"],
            importance_by_level={"global": {"all": {"f1": 0.5}}},
            raw_values_by_level=None,
        )

        assert result.get_raw_values("global", "all") is None

    def test_to_dataframe(self, sample_result: HierarchicalResult) -> None:
        """Test converting to long-form DataFrame."""
        df = sample_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "level" in df.columns
        assert "cohort" in df.columns
        assert "feature" in df.columns
        assert "importance" in df.columns

        global_rows = df[df["level"] == "global"]
        assert len(global_rows) == 3

        state_rows = df[df["level"] == "state"]
        assert len(state_rows) == 6

    def test_repr(self, sample_result: HierarchicalResult) -> None:
        """Test string representation."""
        repr_str = repr(sample_result)

        assert "HierarchicalResult" in repr_str
        assert "levels=3" in repr_str
        assert "features=3" in repr_str
        assert "shap" in repr_str


class TestHierarchicalResultEdgeCases:
    """Tests for edge cases."""

    def test_single_level(self) -> None:
        """Test result with single level."""
        result = HierarchicalResult(
            levels=["global"],
            features=["f1", "f2"],
            importance_by_level={
                "global": {"all": {"f1": 0.6, "f2": 0.4}},
            },
        )

        assert result.get_global()["f1"] == 0.6
        assert len(result.get_cohorts_at_level("global")) == 1

    def test_single_feature(self) -> None:
        """Test result with single feature."""
        result = HierarchicalResult(
            levels=["global", "group"],
            features=["f1"],
            importance_by_level={
                "global": {"all": {"f1": 0.5}},
                "group": {"A": {"f1": 0.4}, "B": {"f1": 0.6}},
            },
        )

        assert result.get_global()["f1"] == 0.5
        top = result.get_top_features("group", "B", n=1)
        assert top[0] == ("f1", 0.6)

    def test_many_cohorts(self) -> None:
        """Test result with many cohorts."""
        n_cohorts = 100
        cohort_data = {
            f"cohort_{i}": {"f1": i / n_cohorts, "f2": 1 - i / n_cohorts} for i in range(n_cohorts)
        }

        result = HierarchicalResult(
            levels=["global", "level1"],
            features=["f1", "f2"],
            importance_by_level={
                "global": {"all": {"f1": 0.5, "f2": 0.5}},
                "level1": cohort_data,
            },
        )

        cohorts = result.get_cohorts_at_level("level1")
        assert len(cohorts) == n_cohorts

        df = result.get_level_df("level1")
        assert len(df) == n_cohorts

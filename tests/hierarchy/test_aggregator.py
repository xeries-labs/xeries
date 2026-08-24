"""Unit tests for HierarchicalAggregator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xeries.core.types import FeatureImportanceResult, SHAPResult
from xeries.hierarchy.aggregator import HierarchicalAggregator
from xeries.hierarchy.definition import HierarchyDefinition
from xeries.hierarchy.types import HierarchicalResult


class TestHierarchicalAggregatorSHAP:
    """Tests for SHAP value aggregation."""

    @pytest.fixture
    def hierarchy(self) -> HierarchyDefinition:
        """Create test hierarchy."""
        return HierarchyDefinition(
            levels=["state", "store"],
            columns=["state_id", "store_id"],
        )

    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        """Create test data with hierarchy columns."""
        return pd.DataFrame(
            {
                "state_id": ["TX", "TX", "TX", "TX", "WI", "WI", "WI", "WI"],
                "store_id": ["S1", "S1", "S2", "S2", "S1", "S1", "S2", "S2"],
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            }
        )

    @pytest.fixture
    def shap_result(self) -> SHAPResult:
        """Create mock SHAP result."""
        return SHAPResult(
            shap_values=np.array(
                [
                    [0.1, 0.2],
                    [0.15, 0.25],
                    [0.3, 0.1],
                    [0.35, 0.15],
                    [0.5, 0.4],
                    [0.55, 0.45],
                    [0.2, 0.6],
                    [0.25, 0.65],
                ]
            ),
            base_values=np.array([1.0] * 8),
            feature_names=["feature_1", "feature_2"],
            data=np.zeros((8, 2)),
        )

    def test_aggregate_shap_global_level(
        self,
        hierarchy: HierarchyDefinition,
        test_data: pd.DataFrame,
        shap_result: SHAPResult,
    ) -> None:
        """Test SHAP aggregation at global level."""
        aggregator = HierarchicalAggregator(hierarchy)
        result = aggregator.aggregate_shap(shap_result, test_data)

        assert isinstance(result, HierarchicalResult)
        assert "global" in result.levels
        assert "all" in result.importance_by_level["global"]

        global_imp = result.get_global()
        assert "feature_1" in global_imp
        assert "feature_2" in global_imp
        assert global_imp["feature_1"] > 0
        assert global_imp["feature_2"] > 0

    def test_aggregate_shap_state_level(
        self,
        hierarchy: HierarchyDefinition,
        test_data: pd.DataFrame,
        shap_result: SHAPResult,
    ) -> None:
        """Test SHAP aggregation at state level."""
        aggregator = HierarchicalAggregator(hierarchy)
        result = aggregator.aggregate_shap(shap_result, test_data)

        assert "state" in result.levels
        cohorts = result.get_cohorts_at_level("state")
        assert "TX" in cohorts
        assert "WI" in cohorts

        tx_imp = result.importance_by_level["state"]["TX"]
        wi_imp = result.importance_by_level["state"]["WI"]
        assert tx_imp["feature_1"] != wi_imp["feature_1"]

    def test_aggregate_shap_store_level(
        self,
        hierarchy: HierarchyDefinition,
        test_data: pd.DataFrame,
        shap_result: SHAPResult,
    ) -> None:
        """Test SHAP aggregation at store level."""
        aggregator = HierarchicalAggregator(hierarchy)
        result = aggregator.aggregate_shap(shap_result, test_data)

        assert "store" in result.levels
        cohorts = result.get_cohorts_at_level("store")
        assert "TX_S1" in cohorts
        assert "TX_S2" in cohorts
        assert "WI_S1" in cohorts
        assert "WI_S2" in cohorts

    def test_aggregate_shap_specific_levels(
        self,
        hierarchy: HierarchyDefinition,
        test_data: pd.DataFrame,
        shap_result: SHAPResult,
    ) -> None:
        """Test aggregation with specific levels only."""
        aggregator = HierarchicalAggregator(hierarchy)
        result = aggregator.aggregate_shap(shap_result, test_data, levels=["state"])

        assert "global" in result.levels
        assert "state" in result.levels
        assert "store" not in result.levels

    def test_aggregate_shap_raw_values(
        self,
        hierarchy: HierarchyDefinition,
        test_data: pd.DataFrame,
        shap_result: SHAPResult,
    ) -> None:
        """Test that raw values are stored when include_raw=True."""
        aggregator = HierarchicalAggregator(hierarchy)
        result = aggregator.aggregate_shap(shap_result, test_data, include_raw=True)

        assert result.raw_values_by_level is not None
        raw_global = result.get_raw_values("global", "all")
        assert raw_global is not None
        assert raw_global.shape == (8, 2)

    def test_aggregate_shap_no_raw_values(
        self,
        hierarchy: HierarchyDefinition,
        test_data: pd.DataFrame,
        shap_result: SHAPResult,
    ) -> None:
        """Test that raw values are not stored when include_raw=False."""
        aggregator = HierarchicalAggregator(hierarchy)
        result = aggregator.aggregate_shap(shap_result, test_data, include_raw=False)

        assert result.raw_values_by_level is None

    def test_aggregate_shap_data_length_mismatch(
        self,
        hierarchy: HierarchyDefinition,
        shap_result: SHAPResult,
    ) -> None:
        """Test that mismatched data length raises ValueError."""
        short_data = pd.DataFrame({"state_id": ["TX"], "store_id": ["S1"], "feature_1": [1.0]})
        aggregator = HierarchicalAggregator(hierarchy)

        with pytest.raises(ValueError, match="Data length"):
            aggregator.aggregate_shap(shap_result, short_data)


class TestHierarchicalAggregatorImportance:
    """Tests for permutation importance aggregation."""

    @pytest.fixture
    def hierarchy(self) -> HierarchyDefinition:
        """Create test hierarchy."""
        return HierarchyDefinition(
            levels=["group"],
            columns=["group_id"],
        )

    @pytest.fixture
    def test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create test data."""
        X = pd.DataFrame(
            {
                "group_id": ["A", "A", "B", "B"],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "feature_2": [0.5, 1.5, 2.5, 3.5],
            }
        )
        y = pd.Series([1.1, 2.1, 3.1, 4.1])
        return X, y

    @pytest.fixture
    def importance_result(self) -> FeatureImportanceResult:
        """Create mock importance result."""
        return FeatureImportanceResult(
            feature_names=["feature_1", "feature_2"],
            importances=np.array([0.5, 0.3]),
            method="permutation",
        )

    def test_aggregate_importance_without_model(
        self,
        hierarchy: HierarchyDefinition,
        test_data: tuple[pd.DataFrame, pd.Series],
        importance_result: FeatureImportanceResult,
    ) -> None:
        """Test importance aggregation without model (uses global values)."""
        X, y = test_data
        aggregator = HierarchicalAggregator(hierarchy)

        result = aggregator.aggregate_importance(importance_result, X, y, model=None)

        assert isinstance(result, HierarchicalResult)
        assert "global" in result.levels
        assert "group" in result.levels
        assert result.method == "permutation"


class TestHierarchicalAggregatorPerSeries:
    """Tests for aggregating pre-computed per-series results."""

    @pytest.fixture
    def hierarchy(self) -> HierarchyDefinition:
        """Create test hierarchy."""
        return HierarchyDefinition(
            levels=["region"],
            parse_pattern=r"(?P<region>\w+)_\d+",
            series_col="series_id",
        )

    @pytest.fixture
    def per_series_results(self) -> dict[str, FeatureImportanceResult]:
        """Create per-series importance results."""
        return {
            "North_1": FeatureImportanceResult(
                feature_names=["f1", "f2"],
                importances=np.array([0.1, 0.2]),
            ),
            "North_2": FeatureImportanceResult(
                feature_names=["f1", "f2"],
                importances=np.array([0.15, 0.25]),
            ),
            "South_1": FeatureImportanceResult(
                feature_names=["f1", "f2"],
                importances=np.array([0.5, 0.3]),
            ),
        }

    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        """Create test data."""
        return pd.DataFrame(
            {
                "series_id": ["North_1", "North_1", "North_2", "South_1"],
                "value": [1, 2, 3, 4],
            }
        )

    def test_aggregate_from_per_series(
        self,
        hierarchy: HierarchyDefinition,
        per_series_results: dict[str, FeatureImportanceResult],
        test_data: pd.DataFrame,
    ) -> None:
        """Test aggregating pre-computed per-series results."""
        aggregator = HierarchicalAggregator(hierarchy)
        result = aggregator.aggregate_from_per_series(per_series_results, test_data)

        assert isinstance(result, HierarchicalResult)
        assert "global" in result.levels
        assert "region" in result.levels

        global_imp = result.get_global()
        assert np.isclose(global_imp["f1"], np.mean([0.1, 0.15, 0.5]))

    def test_aggregate_from_per_series_empty_raises(
        self,
        hierarchy: HierarchyDefinition,
        test_data: pd.DataFrame,
    ) -> None:
        """Test that empty per_series_results raises ValueError."""
        aggregator = HierarchicalAggregator(hierarchy)

        with pytest.raises(ValueError, match="cannot be empty"):
            aggregator.aggregate_from_per_series({}, test_data)

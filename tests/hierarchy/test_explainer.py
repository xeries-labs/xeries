"""Unit tests for HierarchicalExplainer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from xeries.core.types import SHAPResult
from xeries.hierarchy.definition import HierarchyDefinition
from xeries.hierarchy.explainer import HierarchicalExplainer
from xeries.hierarchy.types import HierarchicalResult


class MockSHAPExplainer:
    """Mock SHAP explainer for testing."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self._shap_values: np.ndarray | None = None

    def explain(self, X: pd.DataFrame, **kwargs: Any) -> SHAPResult:
        n_samples = len(X)
        n_features = len(self.feature_names)

        if self._shap_values is not None:
            shap_values = self._shap_values[:n_samples]
        else:
            shap_values = np.random.randn(n_samples, n_features) * 0.1

        return SHAPResult(
            shap_values=shap_values,
            base_values=np.ones(n_samples),
            feature_names=self.feature_names,
            data=X.values,
        )

    def set_shap_values(self, values: np.ndarray) -> None:
        """Set predetermined SHAP values for testing."""
        self._shap_values = values


class TestHierarchicalExplainerInit:
    """Tests for HierarchicalExplainer initialization."""

    def test_init_with_shap_explainer(self) -> None:
        """Test initialization with SHAP explainer."""
        mock_explainer = MockSHAPExplainer(["f1", "f2"])
        hierarchy = HierarchyDefinition(levels=["group"], columns=["group_id"])

        explainer = HierarchicalExplainer(mock_explainer, hierarchy)

        assert explainer.base_explainer is mock_explainer
        assert explainer.hierarchy is hierarchy
        assert explainer._explainer_type == "shap"

    def test_detect_permutation_explainer(self) -> None:
        """Test that permutation explainer type is detected."""
        mock_explainer = MagicMock()
        mock_explainer.__class__.__name__ = "ConditionalPermutationImportance"
        hierarchy = HierarchyDefinition(levels=["group"], columns=["group_id"])

        explainer = HierarchicalExplainer(mock_explainer, hierarchy)

        assert explainer._explainer_type == "permutation"


class TestHierarchicalExplainerExplain:
    """Tests for HierarchicalExplainer.explain()."""

    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        """Create test data."""
        return pd.DataFrame(
            {
                "state_id": ["TX", "TX", "WI", "WI", "TX", "WI"],
                "store_id": ["S1", "S2", "S1", "S2", "S1", "S1"],
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )

    @pytest.fixture
    def hierarchy(self) -> HierarchyDefinition:
        """Create test hierarchy."""
        return HierarchyDefinition(
            levels=["state", "store"],
            columns=["state_id", "store_id"],
        )

    @pytest.fixture
    def mock_shap_explainer(self) -> MockSHAPExplainer:
        """Create mock SHAP explainer."""
        explainer = MockSHAPExplainer(["f1", "f2"])
        explainer.set_shap_values(
            np.array(
                [
                    [0.1, 0.2],
                    [0.15, 0.25],
                    [0.5, 0.3],
                    [0.55, 0.35],
                    [0.12, 0.22],
                    [0.52, 0.32],
                ]
            )
        )
        return explainer

    def test_explain_returns_hierarchical_result(
        self,
        test_data: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test that explain returns HierarchicalResult."""
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)
        result = explainer.explain(test_data)

        assert isinstance(result, HierarchicalResult)
        assert "global" in result.levels
        assert "state" in result.levels
        assert "store" in result.levels

    def test_explain_with_specific_levels(
        self,
        test_data: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test explain with specific levels."""
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)
        result = explainer.explain(test_data, levels=["state"])

        assert "global" in result.levels
        assert "state" in result.levels
        assert "store" not in result.levels

    def test_explain_validates_data(
        self,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test that explain validates data structure."""
        invalid_data = pd.DataFrame({"other": [1, 2], "f1": [1.0, 2.0]})
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)

        with pytest.raises(ValueError, match="Missing required columns"):
            explainer.explain(invalid_data)


class TestHierarchicalExplainerExplainLevel:
    """Tests for HierarchicalExplainer.explain_level()."""

    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        """Create test data."""
        return pd.DataFrame(
            {
                "group_id": ["A", "A", "B", "B"],
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [0.1, 0.2, 0.3, 0.4],
            }
        )

    @pytest.fixture
    def hierarchy(self) -> HierarchyDefinition:
        """Create test hierarchy."""
        return HierarchyDefinition(levels=["group"], columns=["group_id"])

    @pytest.fixture
    def mock_shap_explainer(self) -> MockSHAPExplainer:
        """Create mock SHAP explainer."""
        explainer = MockSHAPExplainer(["f1", "f2"])
        explainer.set_shap_values(np.array([[0.1, 0.2], [0.15, 0.25], [0.5, 0.3], [0.55, 0.35]]))
        return explainer

    def test_explain_level_returns_dataframe(
        self,
        test_data: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test that explain_level returns DataFrame."""
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)
        result = explainer.explain_level(test_data, level="group")

        assert isinstance(result, pd.DataFrame)
        assert "f1" in result.columns
        assert "f2" in result.columns
        assert "A" in result.index or "B" in result.index

    def test_explain_level_specific_cohort(
        self,
        test_data: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test explain_level with specific cohort."""
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)
        result = explainer.explain_level(test_data, level="group", cohort="A")

        assert len(result) == 1
        assert "A" in result.index

    def test_explain_level_invalid_cohort_raises(
        self,
        test_data: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test that invalid cohort raises KeyError."""
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)

        with pytest.raises(KeyError, match="not found at level"):
            explainer.explain_level(test_data, level="group", cohort="X")


class TestHierarchicalExplainerCompare:
    """Tests for comparison methods."""

    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        """Create test data."""
        return pd.DataFrame(
            {
                "group_id": ["A", "A", "A", "B", "B", "B"],
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "f3": [10, 20, 30, 40, 50, 60],
            }
        )

    @pytest.fixture
    def hierarchy(self) -> HierarchyDefinition:
        """Create test hierarchy."""
        return HierarchyDefinition(levels=["group"], columns=["group_id"])

    @pytest.fixture
    def mock_shap_explainer(self) -> MockSHAPExplainer:
        """Create mock SHAP explainer."""
        explainer = MockSHAPExplainer(["f1", "f2", "f3"])
        explainer.set_shap_values(
            np.array(
                [
                    [0.1, 0.2, 0.05],
                    [0.12, 0.22, 0.06],
                    [0.11, 0.21, 0.055],
                    [0.5, 0.3, 0.1],
                    [0.52, 0.32, 0.12],
                    [0.51, 0.31, 0.11],
                ]
            )
        )
        return explainer

    def test_compare_cohorts(
        self,
        test_data: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test comparing cohorts."""
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)
        comparison = explainer.compare_cohorts(test_data, level="group", top_n=2)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "A" in comparison.columns
        assert "B" in comparison.columns

    def test_feature_ranking_stability(
        self,
        test_data: pd.DataFrame,
        hierarchy: HierarchyDefinition,
        mock_shap_explainer: MockSHAPExplainer,
    ) -> None:
        """Test feature ranking stability analysis."""
        explainer = HierarchicalExplainer(mock_shap_explainer, hierarchy)
        stability = explainer.feature_ranking_stability(test_data, level="group", top_n=3)

        assert isinstance(stability, pd.DataFrame)
        assert "mean_rank" in stability.columns
        assert "std_rank" in stability.columns
        assert "min_rank" in stability.columns
        assert "max_rank" in stability.columns


class TestHierarchicalExplainerRepr:
    """Tests for string representation."""

    def test_repr(self) -> None:
        """Test repr format."""
        mock_explainer = MockSHAPExplainer(["f1"])
        hierarchy = HierarchyDefinition(levels=["state", "store"], columns=["state_id", "store_id"])
        explainer = HierarchicalExplainer(mock_explainer, hierarchy)

        repr_str = repr(explainer)

        assert "HierarchicalExplainer" in repr_str
        assert "MockSHAPExplainer" in repr_str
        assert "['state', 'store']" in repr_str

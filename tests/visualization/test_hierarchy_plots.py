"""Unit tests for hierarchy visualization functions."""

from __future__ import annotations

import numpy as np
import pytest

from xeries.hierarchy.types import HierarchicalResult


@pytest.fixture
def sample_result() -> HierarchicalResult:
    """Create sample HierarchicalResult for visualization tests."""
    return HierarchicalResult(
        levels=["global", "state", "store"],
        features=["lag_1", "lag_2", "price", "events", "week"],
        importance_by_level={
            "global": {
                "all": {
                    "lag_1": 0.45,
                    "lag_2": 0.25,
                    "price": 0.15,
                    "events": 0.10,
                    "week": 0.05,
                },
            },
            "state": {
                "TX": {
                    "lag_1": 0.40,
                    "lag_2": 0.30,
                    "price": 0.15,
                    "events": 0.10,
                    "week": 0.05,
                },
                "WI": {
                    "lag_1": 0.50,
                    "lag_2": 0.20,
                    "price": 0.15,
                    "events": 0.10,
                    "week": 0.05,
                },
            },
            "store": {
                "TX_S1": {
                    "lag_1": 0.35,
                    "lag_2": 0.35,
                    "price": 0.15,
                    "events": 0.10,
                    "week": 0.05,
                },
                "TX_S2": {
                    "lag_1": 0.45,
                    "lag_2": 0.25,
                    "price": 0.15,
                    "events": 0.10,
                    "week": 0.05,
                },
                "WI_S1": {
                    "lag_1": 0.55,
                    "lag_2": 0.15,
                    "price": 0.15,
                    "events": 0.10,
                    "week": 0.05,
                },
                "WI_S2": {
                    "lag_1": 0.45,
                    "lag_2": 0.25,
                    "price": 0.15,
                    "events": 0.10,
                    "week": 0.05,
                },
            },
        },
        raw_values_by_level={
            "global": {
                "all": np.random.randn(20, 5) * 0.2,
            },
            "state": {
                "TX": np.random.randn(10, 5) * 0.2,
                "WI": np.random.randn(10, 5) * 0.2,
            },
        },
        method="shap",
    )


@pytest.fixture
def sample_result_no_raw() -> HierarchicalResult:
    """Create sample HierarchicalResult without raw values."""
    return HierarchicalResult(
        levels=["global", "group"],
        features=["f1", "f2"],
        importance_by_level={
            "global": {"all": {"f1": 0.6, "f2": 0.4}},
            "group": {
                "A": {"f1": 0.5, "f2": 0.5},
                "B": {"f1": 0.7, "f2": 0.3},
            },
        },
        raw_values_by_level=None,
        method="permutation",
    )


class TestPlotHierarchyBar:
    """Tests for plot_hierarchy_bar function."""

    def test_plot_global_level(self, sample_result: HierarchicalResult) -> None:
        """Test plotting global level bar chart."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_bar

        fig, ax = plot_hierarchy_bar(sample_result, level="global")

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Global Feature Importance"

    def test_plot_state_level(self, sample_result: HierarchicalResult) -> None:
        """Test plotting state level bar chart."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_bar

        fig, ax = plot_hierarchy_bar(sample_result, level="state", cohort="TX")

        assert fig is not None
        assert "TX" in ax.get_title()

    def test_plot_with_top_n(self, sample_result: HierarchicalResult) -> None:
        """Test plotting with limited features."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_bar

        fig, ax = plot_hierarchy_bar(sample_result, level="global", top_n=3)

        assert fig is not None
        assert len(ax.patches) == 3

    def test_plot_invalid_level_raises(self, sample_result: HierarchicalResult) -> None:
        """Test that invalid level raises KeyError."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_bar

        with pytest.raises(KeyError):
            plot_hierarchy_bar(sample_result, level="invalid")

    def test_plot_invalid_cohort_raises(self, sample_result: HierarchicalResult) -> None:
        """Test that invalid cohort raises KeyError."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_bar

        with pytest.raises(KeyError, match="not found at level"):
            plot_hierarchy_bar(sample_result, level="state", cohort="CA")


class TestPlotHierarchyViolin:
    """Tests for plot_hierarchy_violin function."""

    def test_plot_global_violin(self, sample_result: HierarchicalResult) -> None:
        """Test plotting global violin plot."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_violin

        fig, ax = plot_hierarchy_violin(sample_result, level="global")

        assert fig is not None
        assert ax is not None

    def test_plot_state_violin(self, sample_result: HierarchicalResult) -> None:
        """Test plotting state level violin plot."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_violin

        fig, _ax = plot_hierarchy_violin(sample_result, level="state", cohort="TX")

        assert fig is not None

    def test_plot_violin_no_raw_raises(self, sample_result_no_raw: HierarchicalResult) -> None:
        """Test that missing raw values raises ValueError."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_violin

        with pytest.raises(ValueError, match="Raw values not available"):
            plot_hierarchy_violin(sample_result_no_raw, level="global")


class TestPlotHierarchyComparison:
    """Tests for plot_hierarchy_comparison function."""

    def test_plot_comparison(self, sample_result: HierarchicalResult) -> None:
        """Test plotting cohort comparison."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_comparison

        fig, ax = plot_hierarchy_comparison(sample_result, level="state")

        assert fig is not None
        assert ax is not None

    def test_plot_comparison_specific_feature(self, sample_result: HierarchicalResult) -> None:
        """Test plotting comparison for specific feature."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_comparison

        fig, _ax = plot_hierarchy_comparison(sample_result, level="state", feature="lag_1")

        assert fig is not None

    def test_plot_comparison_normalized(self, sample_result: HierarchicalResult) -> None:
        """Test plotting normalized comparison."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_comparison

        fig, _ax = plot_hierarchy_comparison(sample_result, level="store", normalize=True)

        assert fig is not None


class TestPlotHierarchySummary:
    """Tests for plot_hierarchy_summary function."""

    def test_plot_summary_all_levels(self, sample_result: HierarchicalResult) -> None:
        """Test plotting summary for all levels with grid layout."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_summary

        fig, axes = plot_hierarchy_summary(sample_result)

        assert fig is not None
        assert axes.ndim == 2
        assert axes.shape[0] == 3
        n_cols = axes.shape[1]
        assert n_cols == 4

        visible_per_row = []
        for row in range(axes.shape[0]):
            visible = sum(1 for col in range(n_cols) if axes[row, col].get_visible())
            visible_per_row.append(visible)

        assert visible_per_row[0] == 1
        assert visible_per_row[1] == 2
        assert visible_per_row[2] == 4

    def test_plot_summary_specific_levels(self, sample_result: HierarchicalResult) -> None:
        """Test plotting summary for specific levels."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_summary

        fig, axes = plot_hierarchy_summary(sample_result, levels=["global", "state"])

        assert fig is not None
        assert axes.ndim == 2
        assert axes.shape[0] == 2
        assert axes.shape[1] == 2


class TestPlotHierarchyTree:
    """Tests for plot_hierarchy_tree function."""

    def test_plot_tree(self, sample_result: HierarchicalResult) -> None:
        """Test plotting hierarchy tree."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_tree

        fig, ax = plot_hierarchy_tree(sample_result, feature="lag_1")

        assert fig is not None
        assert ax is not None

    def test_plot_tree_invalid_feature_raises(self, sample_result: HierarchicalResult) -> None:
        """Test that invalid feature raises KeyError."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_tree

        with pytest.raises(KeyError, match="not found"):
            plot_hierarchy_tree(sample_result, feature="invalid")


class TestPlotHierarchyHeatmap:
    """Tests for plot_hierarchy_heatmap function."""

    def test_plot_heatmap(self, sample_result: HierarchicalResult) -> None:
        """Test plotting hierarchy heatmap."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_heatmap

        fig, ax = plot_hierarchy_heatmap(sample_result, level="state")

        assert fig is not None
        assert ax is not None

    def test_plot_heatmap_top_n(self, sample_result: HierarchicalResult) -> None:
        """Test plotting heatmap with limited features."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_heatmap

        fig, _ax = plot_hierarchy_heatmap(sample_result, level="store", top_n=3)

        assert fig is not None

    def test_plot_heatmap_no_annotation(self, sample_result: HierarchicalResult) -> None:
        """Test plotting heatmap without annotations."""
        pytest.importorskip("matplotlib")
        from xeries.visualization.hierarchy_plots import plot_hierarchy_heatmap

        fig, _ax = plot_hierarchy_heatmap(sample_result, level="state", annot=False)

        assert fig is not None

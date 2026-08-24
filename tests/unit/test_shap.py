"""Unit tests for ConditionalSHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from xeries.core.types import SHAPResult
from xeries.importance.shap import ConditionalSHAP


class TestConditionalSHAPInit:
    """Tests for ConditionalSHAP initialization."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, RandomForestRegressor]:
        """Create sample data and fitted model."""
        np.random.seed(42)
        n_samples = 60

        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(["A", "B", "C"], n_samples // 3),
                pd.date_range("2023-01-01", periods=n_samples, freq="h"),
            ],
            names=["level", "date"],
        )

        X = pd.DataFrame(
            {
                "lag_1": np.random.randn(n_samples),
                "lag_2": np.random.randn(n_samples),
                "exog_1": np.random.randn(n_samples),
            },
            index=index,
        )
        y = X["lag_1"] * 0.5 + X["lag_2"] * 0.3 + np.random.randn(n_samples) * 0.1

        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X.values, y.values)

        return X, model

    def test_init_default_params(
        self, sample_data: tuple[pd.DataFrame, RandomForestRegressor]
    ) -> None:
        """Test default initialization parameters."""
        X, model = sample_data
        explainer = ConditionalSHAP(model, X, series_col="level")

        assert explainer.series_col == "level"
        assert explainer.n_background_samples == 100
        assert explainer.explainer_type == "auto"
        assert explainer.background_strategy == "series"

    def test_init_auto_detects_tree_model(
        self, sample_data: tuple[pd.DataFrame, RandomForestRegressor]
    ) -> None:
        """Test that auto detection identifies tree models."""
        X, model = sample_data
        explainer = ConditionalSHAP(model, X, series_col="level")

        assert explainer._is_batch_capable is True

    def test_init_kernel_explainer_type(
        self, sample_data: tuple[pd.DataFrame, RandomForestRegressor]
    ) -> None:
        """Test explicit kernel explainer type."""
        X, model = sample_data
        explainer = ConditionalSHAP(model, X, series_col="level", explainer_type="kernel")

        assert explainer._is_batch_capable is False
        assert len(explainer._series_backgrounds) == 3

    def test_init_global_background_strategy(
        self, sample_data: tuple[pd.DataFrame, RandomForestRegressor]
    ) -> None:
        """Test global background strategy."""
        X, model = sample_data
        explainer = ConditionalSHAP(model, X, series_col="level", background_strategy="global")

        assert explainer.background_strategy == "global"


class TestConditionalSHAPExplain:
    """Tests for ConditionalSHAP explain method."""

    @pytest.fixture
    def explainer_and_data(self) -> tuple[ConditionalSHAP, pd.DataFrame]:
        """Create explainer and test data."""
        np.random.seed(42)
        n_samples = 60

        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(["A", "B", "C"], n_samples // 3),
                pd.date_range("2023-01-01", periods=n_samples, freq="h"),
            ],
            names=["level", "date"],
        )

        X = pd.DataFrame(
            {
                "lag_1": np.random.randn(n_samples),
                "lag_2": np.random.randn(n_samples),
                "exog_1": np.random.randn(n_samples),
            },
            index=index,
        )
        y = X["lag_1"] * 0.5 + X["lag_2"] * 0.3 + np.random.randn(n_samples) * 0.1

        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X.values, y.values)

        explainer = ConditionalSHAP(model, X, series_col="level")

        return explainer, X

    def test_explain_returns_shap_result(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test that explain returns SHAPResult."""
        explainer, X = explainer_and_data
        result = explainer.explain(X.iloc[:10])

        assert isinstance(result, SHAPResult)

    def test_explain_shap_values_shape(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test SHAP values have correct shape."""
        explainer, X = explainer_and_data
        result = explainer.explain(X.iloc[:10])

        assert result.shap_values.shape == (10, 3)
        assert len(result.base_values) == 10
        assert result.feature_names == ["lag_1", "lag_2", "exog_1"]

    def test_explain_tracks_series_ids(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test that series_ids are tracked in result."""
        explainer, X = explainer_and_data
        result = explainer.explain(X)

        assert result.series_ids is not None
        assert len(result.series_ids) == len(X)

    def test_explain_with_explicit_features(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test explain with explicit feature names."""
        explainer, X = explainer_and_data
        result = explainer.explain(X.iloc[:10], feature_names=["lag_1", "lag_2"])

        assert result.shap_values.shape == (10, 2)
        assert result.feature_names == ["lag_1", "lag_2"]

    def test_explain_full_dataset(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test explain on full dataset."""
        explainer, X = explainer_and_data
        result = explainer.explain(X)

        assert result.shap_values.shape == (len(X), 3)
        assert len(result.base_values) == len(X)


class TestConditionalSHAPExplainPerSeries:
    """Tests for ConditionalSHAP explain_per_series method."""

    @pytest.fixture
    def explainer_and_data(self) -> tuple[ConditionalSHAP, pd.DataFrame]:
        """Create explainer and test data."""
        np.random.seed(42)
        n_samples = 60

        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(["A", "B", "C"], n_samples // 3),
                pd.date_range("2023-01-01", periods=n_samples, freq="h"),
            ],
            names=["level", "date"],
        )

        X = pd.DataFrame(
            {
                "lag_1": np.random.randn(n_samples),
                "lag_2": np.random.randn(n_samples),
                "exog_1": np.random.randn(n_samples),
            },
            index=index,
        )
        y = X["lag_1"] * 0.5 + X["lag_2"] * 0.3 + np.random.randn(n_samples) * 0.1

        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X.values, y.values)

        explainer = ConditionalSHAP(model, X, series_col="level")

        return explainer, X

    def test_explain_per_series_returns_dict(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test that explain_per_series returns dictionary."""
        explainer, X = explainer_and_data
        results = explainer.explain_per_series(X)

        assert isinstance(results, dict)
        assert len(results) == 3
        assert set(results.keys()) == {"A", "B", "C"}

    def test_explain_per_series_result_types(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test that each result is a SHAPResult."""
        explainer, X = explainer_and_data
        results = explainer.explain_per_series(X)

        for _series_id, result in results.items():
            assert isinstance(result, SHAPResult)

    def test_explain_per_series_correct_samples(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test that each series has correct number of samples."""
        explainer, X = explainer_and_data
        results = explainer.explain_per_series(X)

        for _series_id, result in results.items():
            assert result.shap_values.shape[0] == 20

    def test_explain_per_series_min_samples(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test min_samples filtering."""
        explainer, X = explainer_and_data
        results = explainer.explain_per_series(X, min_samples=25)

        assert len(results) == 0

    def test_explain_per_series_series_ids_tracked(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test that series_ids are set in each result."""
        explainer, X = explainer_and_data
        results = explainer.explain_per_series(X)

        for series_id, result in results.items():
            assert result.series_ids is not None
            assert all(sid == series_id for sid in result.series_ids)


class TestConditionalSHAPMethods:
    """Tests for ConditionalSHAP utility methods."""

    @pytest.fixture
    def explainer_and_data(self) -> tuple[ConditionalSHAP, pd.DataFrame]:
        """Create explainer and test data."""
        np.random.seed(42)
        n_samples = 60

        index = pd.MultiIndex.from_arrays(
            [
                np.repeat(["A", "B", "C"], n_samples // 3),
                pd.date_range("2023-01-01", periods=n_samples, freq="h"),
            ],
            names=["level", "date"],
        )

        X = pd.DataFrame(
            {
                "lag_1": np.random.randn(n_samples),
                "lag_2": np.random.randn(n_samples),
                "exog_1": np.random.randn(n_samples),
            },
            index=index,
        )
        y = X["lag_1"] * 0.5 + X["lag_2"] * 0.3 + np.random.randn(n_samples) * 0.1

        model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        model.fit(X.values, y.values)

        explainer = ConditionalSHAP(model, X, series_col="level")

        return explainer, X

    def test_explain_instance(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test explain_instance method."""
        explainer, X = explainer_and_data
        instance = X.iloc[0]
        result = explainer.explain_instance(instance)

        assert isinstance(result, SHAPResult)
        assert result.shap_values.shape == (1, 3)

    def test_global_importance(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test global_importance method."""
        explainer, X = explainer_and_data
        importance = explainer.global_importance(X)

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "mean_abs_shap" in importance.columns
        assert len(importance) == 3

    def test_global_importance_with_sampling(
        self, explainer_and_data: tuple[ConditionalSHAP, pd.DataFrame]
    ) -> None:
        """Test global_importance with n_samples parameter."""
        explainer, X = explainer_and_data
        importance = explainer.global_importance(X, n_samples=10)

        assert isinstance(importance, pd.DataFrame)


class TestSHAPResultMethods:
    """Tests for SHAPResult methods."""

    @pytest.fixture
    def sample_result(self) -> SHAPResult:
        """Create sample SHAPResult."""
        np.random.seed(42)
        n_samples = 30

        shap_values = np.random.randn(n_samples, 3)
        base_values = np.random.randn(n_samples)
        feature_names = ["feature_1", "feature_2", "feature_3"]
        data = pd.DataFrame(np.random.randn(n_samples, 3), columns=feature_names)
        series_ids = pd.Series(np.repeat(["A", "B", "C"], n_samples // 3))

        return SHAPResult(
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            data=data,
            series_ids=series_ids,
        )

    def test_to_dataframe(self, sample_result: SHAPResult) -> None:
        """Test to_dataframe method."""
        df = sample_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == sample_result.feature_names
        assert len(df) == len(sample_result.shap_values)

    def test_mean_abs_shap(self, sample_result: SHAPResult) -> None:
        """Test mean_abs_shap method."""
        df = sample_result.mean_abs_shap()

        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns
        assert len(df) == 3

    def test_mean_abs_shap_by_series(self, sample_result: SHAPResult) -> None:
        """Test mean_abs_shap_by_series method."""
        df = sample_result.mean_abs_shap_by_series()

        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == {"A", "B", "C"}
        assert list(df.columns) == sample_result.feature_names

    def test_mean_abs_shap_by_series_raises_without_series_ids(self) -> None:
        """Test that mean_abs_shap_by_series raises without series_ids."""
        result = SHAPResult(
            shap_values=np.random.randn(10, 3),
            base_values=np.random.randn(10),
            feature_names=["f1", "f2", "f3"],
            data=pd.DataFrame(np.random.randn(10, 3)),
            series_ids=None,
        )

        with pytest.raises(ValueError, match="series_ids not set"):
            result.mean_abs_shap_by_series()


class TestExplainerTypeDetection:
    """Tests for explainer type auto-detection."""

    def test_detects_random_forest_as_tree(self) -> None:
        """Test that RandomForest is detected as tree."""
        X = pd.DataFrame({"a": [1, 2, 3], "level": ["A", "A", "A"]})
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X[["a"]], [1, 2, 3])

        explainer = ConditionalSHAP(model, X, series_col="level")
        assert explainer._is_batch_capable is True

    def test_detects_ridge_as_linear(self) -> None:
        """Test that Ridge is detected as linear."""
        X = pd.DataFrame({"a": [1, 2, 3], "level": ["A", "A", "A"]})
        model = Ridge()
        model.fit(X[["a"]], [1, 2, 3])

        explainer = ConditionalSHAP(model, X, series_col="level")
        assert explainer._is_batch_capable is True


class TestBackgroundStrategies:
    """Tests for different background strategies."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, RandomForestRegressor]:
        """Create sample data and model."""
        np.random.seed(42)
        n_samples = 60

        X = pd.DataFrame(
            {
                "lag_1": np.random.randn(n_samples),
                "lag_2": np.random.randn(n_samples),
                "level": np.repeat(["A", "B", "C"], n_samples // 3),
            }
        )
        y = X["lag_1"] * 0.5 + X["lag_2"] * 0.3

        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X[["lag_1", "lag_2"]], y)

        return X, model

    def test_series_background_prepares_per_series(
        self, sample_data: tuple[pd.DataFrame, RandomForestRegressor]
    ) -> None:
        """Test series strategy prepares backgrounds per series."""
        X, model = sample_data
        explainer = ConditionalSHAP(
            model,
            X,
            series_col="level",
            explainer_type="kernel",
            background_strategy="series",
        )

        assert len(explainer._series_backgrounds) == 3
        assert "A" in explainer._series_backgrounds
        assert "B" in explainer._series_backgrounds
        assert "C" in explainer._series_backgrounds

    def test_global_background_prepares_once(
        self, sample_data: tuple[pd.DataFrame, RandomForestRegressor]
    ) -> None:
        """Test global strategy prepares single background."""
        X, model = sample_data
        explainer = ConditionalSHAP(
            model,
            X,
            series_col="level",
            explainer_type="kernel",
            background_strategy="global",
        )

        assert len(explainer._series_backgrounds) == 0

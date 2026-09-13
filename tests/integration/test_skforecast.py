"""Integration tests for skforecast adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skforecast")

from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.ensemble import RandomForestRegressor

from xeries.adapters.skforecast import SkforecastAdapter, from_skforecast
from xeries.importance.permutation import ConditionalPermutationImportance


@pytest.fixture
def sample_series_data() -> pd.DataFrame:
    """Create sample multi-series data in skforecast wide format."""
    rng = np.random.default_rng(42)
    n_periods = 100
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="h")
    data = {}
    for series_id in ["MT_001", "MT_002", "MT_003"]:
        base = rng.normal() * 10
        trend = np.linspace(0, 2, n_periods)
        noise = rng.normal(size=n_periods) * 0.5
        data[series_id] = base + trend + noise
    return pd.DataFrame(data, index=dates)


@pytest.mark.integration
class TestSkforecastIntegration:
    """Integration tests requiring skforecast."""

    def test_adapter_with_forecaster(self, sample_series_data: pd.DataFrame) -> None:
        """Adapter extracts training data from ForecasterRecursiveMultiSeries (0.21+)."""
        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
            lags=5,
        )
        forecaster.fit(series=sample_series_data)

        adapter = SkforecastAdapter(forecaster, series=sample_series_data)
        X, y = adapter.get_training_data()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert isinstance(X.index, pd.MultiIndex) or "_level_skforecast" in X.columns

        feature_names = adapter.get_feature_names()
        assert len(feature_names) == 5
        assert all(name.startswith("lag_") for name in feature_names)
        assert set(adapter.get_series_ids()) == {"MT_001", "MT_002", "MT_003"}
        assert len(adapter.predict(X)) == len(X)

    def test_full_pipeline_with_skforecast(self, sample_series_data: pd.DataFrame) -> None:
        """skforecast adapter plus conditional permutation importance."""
        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
            lags=3,
        )
        forecaster.fit(series=sample_series_data)

        adapter = SkforecastAdapter(forecaster, series=sample_series_data)
        X, y = adapter.get_training_data()

        explainer = ConditionalPermutationImportance(
            model=adapter,
            metric="mse",
            strategy="auto",
            n_repeats=2,
            n_jobs=1,
            random_state=42,
        )
        result = explainer.explain(X, y, features=["lag_1", "lag_2"])

        assert len(result.feature_names) == 2
        assert len(result.importances) == 2
        assert len(result.to_dataframe()) == 2

    def test_from_skforecast_helper(self, sample_series_data: pd.DataFrame) -> None:
        """from_skforecast helper wires lags and series column."""
        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=5, random_state=42),
            lags=3,
        )
        forecaster.fit(series=sample_series_data)
        adapter = from_skforecast(forecaster, series=sample_series_data)

        assert adapter.n_lags == 3
        assert adapter.get_series_column() in ("level", "_level_skforecast")

    def test_get_training_data_passes_series_at_call_time(
        self, sample_series_data: pd.DataFrame
    ) -> None:
        """Series may be omitted from the adapter if passed to get_training_data."""
        forecaster = ForecasterRecursiveMultiSeries(
            estimator=RandomForestRegressor(n_estimators=5, random_state=42),
            lags=3,
        )
        forecaster.fit(series=sample_series_data)
        adapter = SkforecastAdapter(forecaster)
        X, y = adapter.get_training_data(series=sample_series_data)
        assert len(X) == len(y)

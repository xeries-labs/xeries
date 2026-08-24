"""SHAP implementations for time series models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from xeries.core.base import AttributionExplainer
from xeries.core.types import ModelProtocol, SHAPResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

ExplainerType = Literal["tree", "linear", "kernel", "deep", "auto"]
BackgroundStrategy = Literal["series", "global"]


class ConditionalSHAP(AttributionExplainer):
    """Conditional SHAP explainer for multi-series forecasting models.

    This explainer computes SHAP values with support for multiple SHAP explainer
    types and flexible background data strategies:

    - **Tree models** (LightGBM, XGBoost, RandomForest): Uses TreeExplainer for
      fast batch computation without background data.
    - **Other models**: Uses KernelExplainer with series-specific or global
      background data.

    Supports both:
    - `explain()`: Full dataset computation with series tracking
    - `explain_per_series()`: Separate computation for each series

    Example with tree model (auto-detected):
        >>> explainer = ConditionalSHAP(lgb_model, X_train, series_col='level')
        >>> result = explainer.explain(X_test)  # Fast batch computation
        >>> print(result.mean_abs_shap())

    Example with kernel explainer:
        >>> explainer = ConditionalSHAP(
        ...     model, X_train,
        ...     series_col='level',
        ...     explainer_type='kernel',
        ...     background_strategy='series'  # Series-specific backgrounds
        ... )
        >>> result = explainer.explain(X_test)

    Example per-series analysis:
        >>> results = explainer.explain_per_series(X_test)
        >>> for series_id, result in results.items():
        ...     print(f"{series_id}: {result.mean_abs_shap()}")

    Example with hierarchical aggregation:
        >>> from xeries.hierarchy import HierarchyDefinition, HierarchicalExplainer
        >>> hierarchy = HierarchyDefinition(
        ...     levels=['state', 'store'],
        ...     columns=['state_id', 'store_id']
        ... )
        >>> hierarchical = HierarchicalExplainer(explainer, hierarchy)
        >>> result = hierarchical.explain(X_test, include_raw=True)
    """

    SUPPORTED_EXPLAINER_TYPES = ("tree", "linear", "kernel", "deep", "auto")

    def __init__(
        self,
        model: ModelProtocol,
        background_data: pd.DataFrame,
        series_col: str = "level",
        n_background_samples: int = 100,
        explainer_type: ExplainerType = "auto",
        explainer: Any = None,
        background_strategy: BackgroundStrategy = "series",
        random_state: int | None = None,
    ) -> None:
        """Initialize the conditional SHAP explainer.

        Args:
            model: A model with a predict method.
            background_data: Full dataset to sample background from.
            series_col: Column or index level containing series identifiers.
            n_background_samples: Number of background samples per series/cohort.
            explainer_type: Type of SHAP explainer to use:
                - 'auto': Auto-detect based on model type (recommended)
                - 'tree': Use TreeExplainer (fast, for tree-based models)
                - 'kernel': Use KernelExplainer (slow, model-agnostic)
                - 'linear': Use LinearExplainer (for linear models)
                - 'deep': Use DeepExplainer (for neural networks)
            explainer: Pre-created SHAP explainer instance. If provided,
                explainer_type is ignored.
            background_strategy: How to select background data:
                - 'series': Use series-specific background for each row
                - 'global': Use global background sampled from all data
            random_state: Random seed for reproducibility.
        """
        super().__init__(model, background_data, random_state)
        self.series_col = series_col
        self.n_background_samples = n_background_samples
        self.explainer_type = explainer_type
        self.background_strategy = background_strategy
        self._external_explainer = explainer

        self._rng = np.random.default_rng(random_state)
        self._shap: Any = None
        self._explainer: Any = None
        self._series_backgrounds: dict[Any, pd.DataFrame] = {}
        self._global_background: pd.DataFrame | None = None
        self._is_batch_capable = False

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the SHAP library and prepare backgrounds."""
        try:
            import shap

            self._shap = shap
        except ImportError as e:
            raise ImportError(
                "shap package is required for ConditionalSHAP. Install it with: pip install shap"
            ) from e

        if self._external_explainer is not None:
            self._explainer = self._external_explainer
            self._is_batch_capable = self._check_batch_capable(self._explainer)
        else:
            resolved_type = self._resolve_explainer_type()
            self._is_batch_capable = resolved_type in ("tree", "linear", "deep")

            if self._is_batch_capable:
                self._explainer = self._create_batch_explainer(resolved_type)
            else:
                self._prepare_backgrounds()

    def _resolve_explainer_type(self) -> ExplainerType:
        """Resolve 'auto' explainer type to a specific type."""
        if self.explainer_type != "auto":
            return self.explainer_type

        return self._detect_explainer_type(self.model)

    def _detect_explainer_type(self, model: Any) -> ExplainerType:
        """Auto-detect the appropriate explainer type for a model."""
        model_type = type(model).__name__.lower()
        module = type(model).__module__.lower()

        tree_indicators = [
            "lightgbm",
            "xgboost",
            "catboost",
            "randomforest",
            "gradientboosting",
            "extratrees",
            "decisiontree",
            "lgbm",
            "xgb",
            "cb",
        ]

        linear_indicators = [
            "linear",
            "logistic",
            "ridge",
            "lasso",
            "elasticnet",
        ]

        deep_indicators = [
            "keras",
            "tensorflow",
            "torch",
            "neural",
            "mlp",
        ]

        combined = model_type + " " + module

        for indicator in tree_indicators:
            if indicator in combined:
                return "tree"

        for indicator in linear_indicators:
            if indicator in combined:
                return "linear"

        for indicator in deep_indicators:
            if indicator in combined:
                return "deep"

        return "kernel"

    def _check_batch_capable(self, explainer: Any) -> bool:
        """Check if an explainer supports batch computation."""
        explainer_name = type(explainer).__name__.lower()
        return any(t in explainer_name for t in ["tree", "linear", "deep", "gradient"])

    def _create_batch_explainer(self, explainer_type: ExplainerType) -> Any:
        """Create a SHAP explainer for batch computation."""
        shap = self._shap

        if explainer_type == "tree":
            return shap.TreeExplainer(self.model)
        elif explainer_type == "linear":
            bg = self._get_global_background()
            feature_cols = self._infer_feature_names(bg)
            return shap.LinearExplainer(self.model, bg[feature_cols])
        elif explainer_type == "deep":
            bg = self._get_global_background()
            feature_cols = self._infer_feature_names(bg)
            return shap.DeepExplainer(self.model, bg[feature_cols].values)
        else:
            raise ValueError(f"Unsupported batch explainer type: {explainer_type}")

    def _prepare_backgrounds(self) -> None:
        """Prepare background datasets based on strategy."""
        if self.background_strategy == "global":
            self._global_background = self._get_global_background()
        else:
            series_ids = self._get_series_ids(self.background_data)
            unique_series = series_ids.unique()

            for series_id in unique_series:
                mask = series_ids == series_id
                series_data = self.background_data[mask]

                if len(series_data) > self.n_background_samples:
                    indices = self._rng.choice(
                        len(series_data),
                        size=self.n_background_samples,
                        replace=False,
                    )
                    series_data = series_data.iloc[indices]

                self._series_backgrounds[series_id] = series_data

    def _get_global_background(self) -> pd.DataFrame:
        """Get global background data sampled from all series."""
        if self._global_background is not None:
            return self._global_background

        if len(self.background_data) > self.n_background_samples:
            indices = self._rng.choice(
                len(self.background_data),
                size=self.n_background_samples,
                replace=False,
            )
            return self.background_data.iloc[indices]

        return self.background_data

    def _get_series_ids(self, X: pd.DataFrame) -> pd.Series:
        """Extract series identifiers from DataFrame."""
        if isinstance(X.index, pd.MultiIndex) and self.series_col in X.index.names:
            return X.index.get_level_values(self.series_col).to_series(index=X.index)

        if self.series_col in X.columns:
            return X[self.series_col]

        if self.series_col == "level" and "_level_skforecast" in X.columns:
            return X["_level_skforecast"]

        raise KeyError(f"Series column '{self.series_col}' not found in DataFrame columns or index")

    def _infer_feature_names(self, X: pd.DataFrame) -> list[str]:
        """Infer feature names from DataFrame, excluding non-numeric and series columns."""
        exclude_cols = {self.series_col, "_level_skforecast"}
        if self.series_col == "level":
            exclude_cols.add("_level_skforecast")

        feature_names = []
        for col in X.columns:
            if col in exclude_cols:
                continue
            # Exclude non-numeric columns (strings, categories, etc.)
            if not np.issubdtype(X[col].dtype, np.number):
                continue
            feature_names.append(col)

        return feature_names

    def explain(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SHAPResult:  # type: ignore[override]
        """Compute SHAP values for the given instances.

        Uses batch computation for tree-based models (fast) or
        series-specific background for other models.

        Args:
            X: Instances to explain.
            feature_names: Names of features. If None, auto-inferred from X.

        Returns:
            SHAPResult containing SHAP values, base values, and series_ids.
        """
        if feature_names is None:
            feature_names = self._infer_feature_names(X)

        series_ids = self._get_series_ids(X)

        if self._is_batch_capable:
            return self._explain_batch(X, feature_names, series_ids)
        else:
            return self._explain_conditional(X, feature_names, series_ids)

    def _explain_batch(
        self,
        X: pd.DataFrame,
        feature_names: list[str],
        series_ids: pd.Series,
    ) -> SHAPResult:
        """Compute SHAP values in batch (for tree/linear/deep explainers)."""
        X_features = X[feature_names]
        explanation = self._explainer(X_features)

        shap_values = self._extract_shap_values(explanation)
        base_values = self._extract_base_values(explanation, len(X))

        return SHAPResult(
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            data=X,
            series_ids=series_ids.reset_index(drop=True),
        )

    def _explain_conditional(
        self,
        X: pd.DataFrame,
        feature_names: list[str],
        series_ids: pd.Series,
    ) -> SHAPResult:
        """Compute SHAP values with series-specific backgrounds (for kernel)."""
        shap = self._shap
        all_shap_values = []
        all_base_values = []

        unique_series = series_ids.unique()

        for series_id in unique_series:
            mask = series_ids == series_id
            X_series = X.loc[mask]

            background = self._get_background_for_series(series_id)
            background_features = background[feature_names].values

            explainer = shap.KernelExplainer(
                self.model.predict,
                background_features,
            )

            X_features = X_series[feature_names].values
            shap_values = explainer.shap_values(X_features)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = np.atleast_2d(shap_values)
            all_shap_values.append(shap_values)

            base_value = (
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, list | np.ndarray)
                and np.asarray(explainer.expected_value).ndim > 0
                and len(explainer.expected_value) == 1
                else explainer.expected_value
            )
            all_base_values.extend([base_value] * len(X_series))

        combined_shap = np.vstack(all_shap_values)

        reordered_shap = np.zeros_like(combined_shap)
        reordered_base = np.zeros(len(X))

        idx = 0
        for series_id in unique_series:
            mask = series_ids == series_id
            series_indices = np.where(mask)[0]
            n_samples = len(series_indices)

            for i, orig_idx in enumerate(series_indices):
                reordered_shap[orig_idx] = combined_shap[idx + i]
                reordered_base[orig_idx] = all_base_values[idx + i]

            idx += n_samples

        return SHAPResult(
            shap_values=reordered_shap,
            base_values=reordered_base,
            feature_names=feature_names,
            data=X,
            series_ids=series_ids.reset_index(drop=True),
        )

    def _get_background_for_series(self, series_id: Any) -> pd.DataFrame:
        """Get background data for a specific series."""
        if self.background_strategy == "global":
            return self._get_global_background()

        if series_id not in self._series_backgrounds:
            all_backgrounds = pd.concat(list(self._series_backgrounds.values()))
            indices = self._rng.choice(
                len(all_backgrounds),
                size=min(self.n_background_samples, len(all_backgrounds)),
                replace=False,
            )
            return all_backgrounds.iloc[indices]

        return self._series_backgrounds[series_id]

    def _extract_shap_values(self, explanation: Any) -> NDArray[np.floating[Any]]:
        """Extract SHAP values array from explanation object."""
        values = explanation.values

        if isinstance(values, list):
            values = values[0]

        if values.ndim == 3:
            values = values[:, :, 0]

        return np.asarray(values, dtype=np.float64)

    def _extract_base_values(self, explanation: Any, n_samples: int) -> NDArray[np.floating[Any]]:
        """Extract base values from explanation object."""
        base_values = explanation.base_values

        if isinstance(base_values, list | np.ndarray):
            base_values = np.asarray(base_values)
            if base_values.ndim == 0:
                base_values = np.full(n_samples, float(base_values))
            elif base_values.ndim == 2:
                base_values = base_values[:, 0]
        else:
            base_values = np.full(n_samples, float(base_values))

        return base_values.astype(np.float64)

    def explain_instance(
        self,
        instance: pd.Series | pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> SHAPResult:
        """Explain a single instance using series-specific background.

        Args:
            instance: Single instance to explain (Series or single-row DataFrame).
            feature_names: Names of features to include.

        Returns:
            SHAPResult for the single instance.
        """
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T
            if isinstance(instance.index[0], tuple) and isinstance(
                self.background_data.index, pd.MultiIndex
            ):
                instance.index = pd.MultiIndex.from_tuples(
                    instance.index, names=self.background_data.index.names
                )

        return self.explain(instance, feature_names)

    def explain_per_series(
        self,
        X: pd.DataFrame,
        series_col: str | None = None,
        feature_names: list[str] | None = None,
        min_samples: int = 10,
    ) -> dict[Any, SHAPResult]:
        """Compute SHAP values separately for each series.

        This method filters the data by each unique series ID and computes
        SHAP values independently for each series. This is useful for
        detailed per-series analysis.

        Args:
            X: Input features DataFrame.
            series_col: Name of the column or MultiIndex level containing series IDs.
                If None, uses self.series_col.
            feature_names: Names of features to compute SHAP values for.
                If None, auto-inferred from X.
            min_samples: Minimum number of samples required per series.
                Series with fewer samples are skipped.

        Returns:
            Dictionary mapping series IDs to SHAPResult objects.

        Example:
            >>> explainer = ConditionalSHAP(model, X_train, series_col='level')
            >>> results = explainer.explain_per_series(X_test)
            >>> for series_id, result in results.items():
            ...     print(f"{series_id}: {result.mean_abs_shap()}")
        """
        series_col = series_col or self.series_col

        if feature_names is None:
            feature_names = self._infer_feature_names(X)

        series_ids = self._get_series_ids(X)
        unique_series = series_ids.unique()

        results: dict[Any, SHAPResult] = {}

        for series_id in unique_series:
            mask = series_ids == series_id
            X_series = X.loc[mask]

            if len(X_series) < min_samples:
                continue

            if self._is_batch_capable:
                result = self._explain_series_batch(X_series, feature_names, series_id)
            else:
                result = self._explain_series_conditional(X_series, feature_names, series_id)

            results[series_id] = result

        return results

    def _explain_series_batch(
        self,
        X: pd.DataFrame,
        feature_names: list[str],
        series_id: Any,
    ) -> SHAPResult:
        """Compute SHAP values for a single series using batch explainer."""
        X_features = X[feature_names]
        explanation = self._explainer(X_features)

        shap_values = self._extract_shap_values(explanation)
        base_values = self._extract_base_values(explanation, len(X))

        series_ids = pd.Series([series_id] * len(X))

        return SHAPResult(
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            data=X,
            series_ids=series_ids,
        )

    def _explain_series_conditional(
        self,
        X: pd.DataFrame,
        feature_names: list[str],
        series_id: Any,
    ) -> SHAPResult:
        """Compute SHAP values for a single series using series-specific background."""
        shap = self._shap

        background = self._get_background_for_series(series_id)
        background_features = background[feature_names].values

        explainer = shap.KernelExplainer(
            self.model.predict,
            background_features,
        )

        X_features = X[feature_names].values
        shap_values = explainer.shap_values(X_features)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.atleast_2d(shap_values)

        base_value = (
            explainer.expected_value[0]
            if isinstance(explainer.expected_value, list | np.ndarray)
            and np.asarray(explainer.expected_value).ndim > 0
            and len(explainer.expected_value) == 1
            else explainer.expected_value
        )
        base_values = np.full(len(X), float(base_value))

        series_ids = pd.Series([series_id] * len(X))

        return SHAPResult(
            shap_values=shap_values,
            base_values=base_values,
            feature_names=feature_names,
            data=X,
            series_ids=series_ids,
        )

    def global_importance(
        self,
        X: pd.DataFrame,
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Compute global feature importance from SHAP values.

        Args:
            X: Dataset to compute global importance over.
            n_samples: Number of samples to use. If None, uses all.

        Returns:
            DataFrame with mean absolute SHAP values per feature.
        """
        if n_samples is not None and n_samples < len(X):
            indices = self._rng.choice(len(X), size=n_samples, replace=False)
            X = X.iloc[indices]

        result = self.explain(X)
        return result.mean_abs_shap()

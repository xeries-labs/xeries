"""Visualization utilities for hierarchical feature importance results.

These functions provide visualizations for HierarchicalResult objects,
matching the plots from the hierarchical demand forecasting article:
- Bar plots of mean absolute importance (Figures 4a, 5a)
- Violin plots for SHAP value distributions (Figure 4b)
- Grouped bar comparisons across cohorts (Figures 6b, 7b)
- Multi-panel summary views (Figures 5-7 layout)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from xeries.hierarchy.types import HierarchicalResult


def plot_hierarchy_bar(
    result: HierarchicalResult,
    level: str = "global",
    cohort: str | None = None,
    top_n: int | None = 15,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    color: str = "#1f77b4",
) -> tuple[Figure, Axes]:
    """Plot mean absolute importance as a horizontal bar chart.

    Creates a bar plot showing feature importance at a specific hierarchy
    level, matching Figure 4a and 5a from the article.

    Args:
        result: HierarchicalResult from HierarchicalExplainer.
        level: Hierarchy level to plot ('global' or a defined level).
        cohort: Specific cohort at the level. If None and level is not 'global',
            shows the first cohort at that level.
        top_n: Number of top features to display. If None, shows all.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size (width, height) in inches.
        title: Plot title. If None, auto-generates based on level/cohort.
        color: Bar color.

    Returns:
        Tuple of (Figure, Axes).

    Example:
        >>> result = explainer.explain(X_test)
        >>> fig, ax = plot_hierarchy_bar(result, level="global")
        >>> fig, ax = plot_hierarchy_bar(result, level="state", cohort="TX")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    if level == "global":
        importance = result.get_global()
        cohort_name = "all"
    else:
        cohorts = result.get_cohorts_at_level(level)
        if cohort is None:
            cohort = cohorts[0]
        if cohort not in cohorts:
            raise KeyError(f"Cohort '{cohort}' not found at level '{level}'")
        importance = result.importance_by_level[level][cohort]
        cohort_name = cohort

    df = pd.DataFrame(
        {"feature": list(importance.keys()), "importance": list(importance.values())}
    ).sort_values("importance", ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["importance"].values, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|" if result.method == "shap" else "Importance")

    if title is None:
        if level == "global":
            title = "Global Feature Importance"
        else:
            title = f"Feature Importance: {level.title()} = {cohort_name}"

    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_hierarchy_violin(
    result: HierarchicalResult,
    level: str = "global",
    cohort: str | None = None,
    features: list[str] | None = None,
    top_n: int | None = 10,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str | None = None,
    max_display_points: int = 500,
    cmap: str = "coolwarm",
    use_shap_plot: bool = True,
) -> tuple[Figure, Axes]:
    """Plot SHAP value distribution in SHAP library style.

    Creates a beeswarm-style plot showing the distribution of SHAP values
    for each feature, colored by feature value (blue=low, red=high),
    similar to SHAP library's summary_plot.

    Args:
        result: HierarchicalResult with raw_values_by_level (requires include_raw=True).
        level: Hierarchy level to plot.
        cohort: Specific cohort at the level.
        features: List of features to include. If None, uses top_n by importance.
        top_n: Number of top features if features is None.
        ax: Matplotlib axes to plot on.
        figsize: Figure size (width, height) in inches.
        title: Plot title.
        max_display_points: Maximum points to display per feature (for performance).
        cmap: Colormap for feature values (default: coolwarm like SHAP).
        use_shap_plot: If True and shap is available, use shap.summary_plot directly.

    Returns:
        Tuple of (Figure, Axes).

    Raises:
        ValueError: If raw values are not available in the result.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    if result.raw_values_by_level is None:
        raise ValueError(
            "Raw values not available. Use include_raw=True when computing explanations."
        )

    if level == "global":
        cohort_name = "all"
    else:
        cohorts = result.get_cohorts_at_level(level)
        if cohort is None:
            cohort = cohorts[0]
        cohort_name = cohort

    raw_values = result.get_raw_values(level, cohort_name)
    if raw_values is None:
        raise ValueError(f"No raw values found for level='{level}', cohort='{cohort_name}'")

    if features is None:
        importance = result.importance_by_level[level][cohort_name]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            sorted_features = sorted_features[:top_n]
        features = [f for f, _ in sorted_features]

    feature_indices = [result.features.index(f) for f in features if f in result.features]
    features = [result.features[i] for i in feature_indices]
    shap_values = raw_values[:, feature_indices]

    feature_data = None
    if result.feature_values_by_level is not None and level in result.feature_values_by_level:
        cohort_features = result.feature_values_by_level[level].get(cohort_name)
        if cohort_features is not None:
            feature_data = cohort_features[:, feature_indices]

    if use_shap_plot:
        try:
            import shap

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()

            plt.sca(ax)
            shap.summary_plot(
                shap_values,
                feature_names=features,
                features=feature_data,
                max_display=len(features),
                show=False,
                plot_type="violin",
            )

            if title is None:
                if level == "global":
                    title = "SHAP Value Distribution (Global)"
                else:
                    title = f"SHAP Value Distribution: {level.title()} = {cohort_name}"

            ax.set_title(title)
            return fig, ax

        except ImportError:
            pass

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colormap = plt.get_cmap(cmap)
    n_features = len(features)

    for i in range(n_features):
        shap_vals = shap_values[:, i]

        n_points = len(shap_vals)
        if n_points > max_display_points:
            indices = np.random.choice(n_points, max_display_points, replace=False)
            shap_vals_plot = shap_vals[indices]
            feat_vals_plot = feature_data[indices, i] if feature_data is not None else None
        else:
            shap_vals_plot = shap_vals
            feat_vals_plot = feature_data[:, i] if feature_data is not None else None

        y_jitter = np.random.normal(0, 0.1, len(shap_vals_plot))
        y_positions = i + y_jitter

        if feat_vals_plot is not None:
            norm = Normalize(vmin=np.nanmin(feat_vals_plot), vmax=np.nanmax(feat_vals_plot))
            colors = colormap(norm(feat_vals_plot))
        else:
            colors = colormap(0.5 * np.ones(len(shap_vals_plot)))

        ax.scatter(
            shap_vals_plot,
            y_positions,
            c=colors,
            s=10,
            alpha=0.6,
            rasterized=len(shap_vals_plot) > 200,
        )

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)

    if feature_data is not None:
        sm = ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("Feature value", rotation=270, labelpad=15)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Low", "", "High"])

    if title is None:
        if level == "global":
            title = "SHAP Value Distribution (Global)"
        else:
            title = f"SHAP Value Distribution: {level.title()} = {cohort_name}"

    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_hierarchy_comparison(
    result: HierarchicalResult,
    level: str,
    feature: str | None = None,
    top_n: int = 10,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (12, 6),
    title: str | None = None,
    normalize: bool = False,
) -> tuple[Figure, Axes]:
    """Compare feature importance across cohorts at the same level.

    Creates a grouped bar chart comparing importance across cohorts,
    matching Figures 6b and 7b from the article.

    Args:
        result: HierarchicalResult from HierarchicalExplainer.
        level: Hierarchy level to compare across cohorts.
        feature: Specific feature to compare. If None, shows top_n features.
        top_n: Number of top features to include if feature is None.
        ax: Matplotlib axes to plot on.
        figsize: Figure size (width, height) in inches.
        title: Plot title.
        normalize: If True, normalize importance within each cohort.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    df = result.get_level_df(level)

    if normalize:
        df = df.div(df.sum(axis=1), axis=0)

    if feature is not None:
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' not found in results")
        plot_data = df[[feature]]
    else:
        mean_importance = df.mean(axis=0).sort_values(ascending=False)
        top_features = mean_importance.head(top_n).index.tolist()
        plot_data = df[top_features]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_cohorts = len(plot_data)
    n_features = len(plot_data.columns)
    x = np.arange(n_features)
    bar_width = 0.8 / max(n_cohorts, 1)

    colors = plt.colormaps["tab10"](np.linspace(0, 1, n_cohorts))

    for idx, (cohort, row) in enumerate(plot_data.iterrows()):
        offset = (idx - (n_cohorts - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            row.values,
            width=bar_width,
            label=str(cohort),
            color=colors[idx],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.columns, rotation=35, ha="right")
    ax.set_ylabel("Importance" if not normalize else "Normalized Importance")
    ax.legend(title=level.title(), bbox_to_anchor=(1.02, 1), loc="upper left")

    if title is None:
        title = f"Feature Importance Comparison by {level.title()}"
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_hierarchy_summary(
    result: HierarchicalResult,
    levels: list[str] | None = None,
    top_n: int = 10,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> tuple[Figure, Any]:
    """Create multi-panel grid showing importance for each cohort at all levels.

    Creates a grid layout with:
    - Rows: one per hierarchy level (global, state, store, etc.)
    - Columns: one per cohort at that level, left-aligned

    This matches the cascading layout from Figures 5-7 in the article,
    showing individual cohort importance rather than level means.

    Args:
        result: HierarchicalResult from HierarchicalExplainer.
        levels: Which levels to include. If None, includes all.
        top_n: Number of top features per panel.
        figsize: Figure size. If None, auto-calculated based on grid dimensions.
        title: Overall figure title.

    Returns:
        Tuple of (Figure, 2D array of Axes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    if levels is None:
        levels = result.levels

    n_rows = len(levels)
    cohorts_per_level: list[list[str]] = []
    for level in levels:
        if level == "global":
            cohorts_per_level.append(["all"])
        else:
            cohorts_per_level.append(result.get_cohorts_at_level(level))

    n_cols = max(len(c) for c in cohorts_per_level)

    if figsize is None:
        figsize = (4 * n_cols, int(3.5 * n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row_idx, level in enumerate(levels):
        cohorts = cohorts_per_level[row_idx]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(cohorts):
                cohort = cohorts[col_idx]
                importance = result.importance_by_level[level][cohort]

                sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                if top_n:
                    sorted_imp = sorted_imp[:top_n]

                features = [f for f, _ in sorted_imp]
                values = [v for _, v in sorted_imp]

                y_pos = np.arange(len(features))
                ax.barh(y_pos, values, color="#1f77b4", alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.invert_yaxis()
                ax.set_xlabel("Mean |SHAP|" if result.method == "shap" else "Importance")
                ax.set_title(f"{level.title()}: {cohort}")
            else:
                ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig, axes


def plot_hierarchy_tree(
    result: HierarchicalResult,
    feature: str,
    figsize: tuple[int, int] = (14, 10),
    cmap: str = "YlOrRd",
    title: str | None = None,
) -> tuple[Figure, Any]:
    """Tree-style visualization of feature importance through hierarchy.

    Creates a novel visualization showing how a single feature's importance
    varies as you descend through the hierarchy levels.

    Args:
        result: HierarchicalResult from HierarchicalExplainer.
        feature: The feature to visualize.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap for importance values.
        title: Plot title.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.patches import FancyBboxPatch
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    if feature not in result.features:
        raise KeyError(f"Feature '{feature}' not found in results")

    levels = result.levels
    n_levels = len(levels)

    fig, ax = plt.subplots(figsize=figsize)

    all_importances = []
    for level in levels:
        for _cohort, feat_imp in result.importance_by_level[level].items():
            all_importances.append(feat_imp.get(feature, 0))

    if all_importances:
        min_imp, max_imp = min(all_importances), max(all_importances)
        norm_range = max_imp - min_imp if max_imp > min_imp else 1
    else:
        min_imp, max_imp, norm_range = 0, 1, 1

    cmap_obj = plt.get_cmap(cmap)

    level_y = {}
    level_cohorts = {}

    for level_idx, level in enumerate(levels):
        y = 1 - (level_idx / (n_levels - 1)) if n_levels > 1 else 0.5
        level_y[level] = y

        cohorts = list(result.importance_by_level[level].keys())
        level_cohorts[level] = cohorts
        n_cohorts = len(cohorts)

        for cohort_idx, cohort in enumerate(cohorts):
            x = (cohort_idx + 1) / (n_cohorts + 1)

            importance = result.importance_by_level[level][cohort].get(feature, 0)
            norm_imp = (importance - min_imp) / norm_range

            color = cmap_obj(norm_imp)

            box = FancyBboxPatch(
                (x - 0.06, y - 0.04),
                0.12,
                0.08,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor="black",
                linewidth=1,
            )
            ax.add_patch(box)

            display_name = cohort if len(cohort) < 12 else cohort[:10] + "..."
            ax.text(
                x,
                y,
                f"{display_name}\n{importance:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([level_y[lv] for lv in levels])
    ax.set_yticklabels([lv.title() for lv in levels])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    sm = ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=min_imp, vmax=max_imp))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Importance")

    if title is None:
        title = f"Hierarchical Importance: {feature}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig, ax


def plot_hierarchy_heatmap(
    result: HierarchicalResult,
    level: str,
    top_n: int | None = 15,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (12, 8),
    cmap: str = "YlOrRd",
    annot: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot feature importance heatmap across cohorts at a level.

    Args:
        result: HierarchicalResult from HierarchicalExplainer.
        level: Hierarchy level to visualize.
        top_n: Number of top features to include.
        ax: Matplotlib axes to plot on.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap name.
        annot: Whether to annotate cells with values.
        title: Plot title.

    Returns:
        Tuple of (Figure, Axes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    df = result.get_level_df(level)

    if top_n is not None:
        mean_importance = df.mean(axis=0).sort_values(ascending=False)
        top_features = mean_importance.head(top_n).index.tolist()
        df = df[top_features]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(df.values, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)

    if annot:
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                value = df.iloc[i, j]
                text_color = "white" if value > df.values.max() * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, label="Importance")

    if title is None:
        title = f"Feature Importance by {level.title()}"
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax

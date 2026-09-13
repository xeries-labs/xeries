# API Reference

This page is a map of the public surface. Full signatures live on the module pages.

## Explainers

- [ConditionalPermutationImportance](importance.md) — cs-PFI (`explain`)
- [ConditionalSHAP](importance.md) — series-aware SHAP (`explain`, `explain_per_series`)
- [HierarchicalExplainer](hierarchy.md) — aggregate explanations by cohort

## Partitioners

- [ManualPartitioner](partitioners.md)
- [TreePartitioner](partitioners.md)

## Adapters

- [from_skforecast](adapters.md) / [SkforecastAdapter](adapters.md)
- [SklearnAdapter](adapters.md)

## Result types

- `FeatureImportanceResult` and `SHAPResult` — see [Importance](importance.md)
- `HierarchicalResult` — see [Hierarchy](hierarchy.md)

## Visualization

- [Importance, SHAP, and hierarchy plots](visualization.md)

## Planned APIs

- SHAP-IQ (`ConditionalSHAPIQ`)
- Feature dropping
- Causal feature importance
- Darts adapter

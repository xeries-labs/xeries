# `specs/`

Per-feature Spec-Driven Development artefacts whose **Home repo** is `xeries`.
The program-wide roadmap lives in `.specify/memory/roadmap.md` (governance
submodule); this folder hosts only the specs assigned to this repo.

## Layout

Each feature gets its own numbered folder, mirroring Spec Kit 0.8 conventions:

```
specs/NNN-short-slug/
├── spec.md          # business-facing specification (REQUIRED)
├── plan.md          # implementation plan + Constitution Check (REQUIRED for new specs)
├── research.md      # Phase 0 research notes (optional)
├── data-model.md    # Phase 1 data-model design (optional)
├── tasks.md         # actionable task list
└── contracts/       # contract tests / interface definitions (optional)
```

Use the templates from the governance submodule:

```bash
cp .specify/templates/spec-template.md  specs/NNN-…/spec.md
cp .specify/templates/plan-template.md  specs/NNN-…/plan.md
cp .specify/templates/tasks-template.md specs/NNN-…/tasks.md
```

## Backfilled vs. new specs

This repo contains both kinds:

- **Backfilled specs** (001 – 007) describe code that was already in `main`
  before SDD adoption. They carry the
  `Status: Backfilled from implementation` banner and pin a commit SHA.
  Mechanical refactors that preserve the public surface do not need a new
  spec but should cite the affected backfilled spec(s) in the commit
  message.
- **New specs** (008+) follow the full Spec Kit flow: `spec.md` → `plan.md`
  (with the Constitution Check gate) → `tasks.md` → implementation.

## Backfilled queue (Home repo: xeries)

| Spec ID | Title |
| --- | --- |
| `001-core-contracts` | `xeries.core` — `BaseExplainer`, `BasePartitioner`, `ModelProtocol`, result dataclasses, type aliases. |
| `002-partitioners` | `xeries.partitioners` — `ManualPartitioner`, `TreePartitioner` (cs-PFI). |
| `003-conditional-permutation-importance` | `xeries.importance.permutation` — `ConditionalPermutationImportance`. |
| `004-framework-adapters` | `xeries.adapters` — `BaseAdapter`, `SklearnAdapter`, `SkforecastAdapter`, `from_skforecast`. |
| `005-conditional-shap` | `xeries.importance.shap` — `ConditionalSHAP` (Tree/Kernel/Linear/Deep auto-detect). |
| `006-hierarchical-explainer` | `xeries.hierarchy` — `HierarchyDefinition`, `HierarchicalAggregator`, `HierarchicalExplainer`, `HierarchicalResult`. |
| `007-visualization` | `xeries.visualization` — `plot_importance_*`, `plot_shap_*`, `plot_hierarchy_*`. |

## Active queue (Home repo: xeries)

| Spec ID | Title | Status |
| --- | --- | --- |
| `008-shapiq-explainer` | `ConditionalSHAPIQ` — any-order Shapley **interactions** via `shapiq` (TreeSHAP-IQ for tree ensembles, fallback `TabularExplainer`). | Planned (active — article-driven) |

See `.specify/memory/roadmap.md` for the full program roadmap, including
specs whose Home repo is `xeries-bench` (e.g. `009-interaction-benchmark`).

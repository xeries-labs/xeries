# AGENTS.md — xeries

> **Read this first.** Whether you are GitHub Copilot, Cursor, Claude, or any
> other agent, this file is the entry point for working on `xeries`. The
> repository follows **Spec-Driven Development (SDD)** under the
> [`xeries-labs/xeries-governance`](https://github.com/xeries-labs/xeries-governance)
> program. The constitution at `.specify/memory/constitution.md` is binding;
> this file just tells you how to find your way around.

## TL;DR for agents

1. **Before writing code, read the spec.** Every change under `src/`, `tests/`,
   or `docs/` MUST trace to a spec in `specs/NNN-<slug>/`. If no spec exists
   for what the user is asking, **author one first** using
   `.specify/templates/spec-template.md` (and update
   `.specify/memory/roadmap.md` via a PR to the governance repo, see below).
2. **Constitution gates are non-negotiable.** Run through the Constitution
   Check table in any new `plan.md` (template: `.specify/templates/plan-template.md`)
   before opening Phase 0 research. A "Needs justification" without a Complexity
   Tracking entry = the gate fails.
3. **Public surface is typed.** `ty check src` and `ruff check src tests` MUST
   pass. New public symbols require type annotations on every parameter and
   return.
4. **Test-first for new contracts.** Add a failing contract test BEFORE the
   implementation commit when you introduce a new public symbol.
5. **Stay in your repo.** This is the **xeries** library. Forecasting model
   training, SHAP-IQ benchmarks, and large data files belong in
   [`xeries-labs/xeries-bench`](https://github.com/xeries-labs/xeries-bench),
   not here. See the Repo-scope rules below.

## Where things live

```
xeries/
├── .specify/                 # → submodule: xeries-labs/xeries-governance @ v1.0.0
│   ├── memory/
│   │   ├── constitution.md   # binding program rules
│   │   ├── roadmap.md        # program-wide roadmap (xeries + xeries-bench)
│   │   └── roadmap-initial.md# frozen historical snapshot
│   └── templates/{spec,plan,tasks}-template.md
├── .github/
│   ├── prompts/              # Copilot slash-command shims (mirror governance)
│   ├── agents/               # Copilot agent definitions
│   ├── copilot-instructions.md
│   └── workflows/            # CI/CD (ci.yml, docs.yml, release.yml)
├── .cursor/
│   ├── rules/specify-rules.mdc  # Cursor always-on rule
│   └── skills/speckit-*/SKILL.md # Cursor Agent slash-command skills (mirror governance)
├── specs/NNN-<slug>/         # per-feature SDD artefacts (Home repo: xeries)
├── src/xeries/               # the library (see public-surface map below)
├── tests/                    # unit + integration tests
├── docs/                     # MkDocs site (mkdocs.yml at root)
├── examples/                 # runnable notebooks / scripts
└── pyproject.toml            # uv-managed; hatchling backend
```

## Public surface map

These are the modules under `src/xeries/`. Each one corresponds to a Backfilled
spec; consult the spec before changing the public contracts.

| Module | What it owns | Backfilled spec |
| --- | --- | --- |
| `xeries.core` | `BaseExplainer`, `BasePartitioner`, `ModelProtocol`, `FeatureImportanceResult`, type aliases (`ArrayLike`, `GroupLabels`) | `specs/001-core-contracts` |
| `xeries.partitioners` | `ManualPartitioner`, `TreePartitioner` | `specs/002-partitioners` |
| `xeries.importance.permutation` | `ConditionalPermutationImportance` | `specs/003-conditional-permutation-importance` |
| `xeries.adapters` | `BaseAdapter`, `SklearnAdapter`, `SkforecastAdapter`, `from_skforecast` | `specs/004-framework-adapters` |
| `xeries.importance.shap` | `ConditionalSHAP` (Tree/Kernel auto-detect) | `specs/005-conditional-shap` |
| `xeries.hierarchy` | `HierarchyDefinition`, `HierarchicalAggregator`, `HierarchicalExplainer`, `HierarchicalResult` | `specs/006-hierarchical-explainer` |
| `xeries.visualization` | `plot_importance_*`, `plot_shap_*`, `plot_hierarchy_*` | `specs/007-visualization` |

The full re-export list is in `src/xeries/__init__.py`. Anything not listed
there is **internal** and may change without a spec.

## Slash commands

The same Spec Kit 0.8 workflow is wired in for two agent back-ends. Pick
whichever your IDE uses; both target the same governance content under
`.specify/`.

### GitHub Copilot — dotted invocation `/speckit.<cmd>`

The `.github/prompts/*.prompt.md` + `.github/agents/*.agent.md` shims map
the slash commands onto this repo:

| Command | Purpose |
| --- | --- |
| `/speckit.constitution` | View / discuss the program constitution. |
| `/speckit.specify` | Author a new `spec.md` from `.specify/templates/spec-template.md`. |
| `/speckit.clarify` | Resolve ambiguities in an existing spec. |
| `/speckit.plan` | Produce `plan.md` from a spec — runs the Constitution Check gate. |
| `/speckit.tasks` | Break a plan into actionable tasks. |
| `/speckit.implement` | Execute the tasks. |
| `/speckit.checklist` | Quality checklist for a spec / plan. |
| `/speckit.analyze` | Cross-artefact consistency audit. |

### Cursor Agent — hyphenated invocation `/speckit-<cmd>`

The `.cursor/skills/speckit-<cmd>/SKILL.md` shims (Spec Kit 0.8.2
`SkillsIntegration` layout) expose the same workflow with hyphen
separators:

| Command | Purpose |
| --- | --- |
| `/speckit-constitution` | View / discuss the program constitution. |
| `/speckit-specify` | Author a new `spec.md`. |
| `/speckit-clarify` | Resolve ambiguities in an existing spec. |
| `/speckit-plan` | Produce `plan.md` — runs the Constitution Check gate. |
| `/speckit-tasks` | Break a plan into actionable tasks. |
| `/speckit-implement` | Execute the tasks. |
| `/speckit-checklist` | Quality checklist for a spec / plan. |
| `/speckit-analyze` | Cross-artefact consistency audit. |

`.cursor/rules/specify-rules.mdc` is always-on and points the agent at the
current plan as additional context.

Other agents (Claude Code, Aider, Continue, …) can simply read the
corresponding `.github/agents/*.agent.md` or `.cursor/skills/*/SKILL.md`
files directly — both describe the same workflows in agent-agnostic
Markdown.

## Repo-scope rules (binding — see constitution)

Do **NOT**, on this repo:

- Add `lightgbm`, `catboost`, or any heavy GBM as a runtime dep. They live in
  `xeries-bench`. (`xeries[shapiq]` is fine; `shapiq` itself is an optional
  extra here.)
- Commit large data files, parquet/feather/npz/h5 artefacts, or rendered
  notebooks with embedded figures > a few hundred KB.
- Edit anything under `.specify/`. That folder is governed by
  `xeries-labs/xeries-governance` and edits MUST go through a PR there. If you
  need to upgrade governance, run:

  ```bash
  git -C .specify fetch --tags
  git -C .specify checkout v1.1.0   # next governance release
  git add .specify
  git commit -m "chore(sdd): bump xeries-governance to v1.1.0"
  ```

- Introduce agent-specific magic in `src/`, `tests/`, or `docs/`. Cursor-only
  comments, Copilot-only prompts, or any other agent-coupled artefact MUST
  live under per-agent directories (e.g. `.github/`, `.cursor/`) and never
  under the runtime surface.

## Development commands

```bash
# Install with the governance submodule.
git clone --recurse-submodules https://github.com/xeries-labs/xeries.git
cd xeries
uv sync --all-extras --group dev

# Lint, format, type-check, test.
uv run ruff format src tests
uv run ruff check src tests
uv run ty check src
uv run pytest

# Docs (strict — broken links / missing pages fail the build).
uv run mkdocs build --strict
uv run mkdocs serve   # local preview at http://127.0.0.1:8000
```

## When in doubt

- **Repo-scope question?** → `.specify/memory/constitution.md` § Repo scope.
- **Should this be a new spec?** → Yes, if it adds, removes, or breaks any
  symbol in `__all__` of any `xeries.*` module. Mechanical refactors that
  preserve the public surface do not need a new spec, but should still cite
  the affected Backfilled spec(s) in the commit message.
- **Want to change the constitution itself?** → Open a PR against
  `xeries-labs/xeries-governance`, not here. After the new governance version
  is tagged (e.g. `v1.1.0`), bump the submodule pointer in this repo as a
  separate commit.

## License

MIT. See [`LICENSE`](./LICENSE).

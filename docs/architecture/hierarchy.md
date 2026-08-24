# Hierarchy Module

The hierarchy module provides tools for defining hierarchical structures and aggregating feature importance across multiple levels.

## Location

`xeries/hierarchy/`

## Architecture

```mermaid
classDiagram
    class HierarchyDefinition {
        +levels: list~str~
        +columns: list~str~
        +parse_pattern: str
        +mapping: dict
        +get_cohorts(X, level) dict
        +validate_data(X)
        +add_hierarchy_columns(X)
    }

    class HierarchicalAggregator {
        +hierarchy: HierarchyDefinition
        +aggregate_shap(result, X) HierarchicalResult
        +aggregate_importance(result, X, y) HierarchicalResult
    }

    class HierarchicalExplainer {
        +base_explainer: BaseExplainer
        +hierarchy: HierarchyDefinition
        +explain(X, y) HierarchicalResult
        +explain_level(X, level, cohort)
        +explain_per_cohort(X, level)
        +compare_cohorts(X, level)
        +feature_ranking_stability(X, level)
    }

    class HierarchicalResult {
        +levels: list~str~
        +features: list~str~
        +importance_by_level: dict
        +raw_values_by_level: dict
        +feature_values_by_level: dict
        +method: str
        +get_level_df(level) DataFrame
        +get_global() dict
        +get_cohorts_at_level(level) list
        +get_raw_values(level, cohort)
    }

    HierarchyDefinition --> HierarchicalAggregator
    HierarchicalAggregator --> HierarchicalExplainer
    HierarchicalExplainer --> HierarchicalResult
```

## Components

### HierarchyDefinition (`definition.py`)

Defines the hierarchical structure for multi-series data.

**Strategies:**

```mermaid
flowchart TB
    subgraph Columns [Column-Based]
        C1[state_id column]
        C2[store_id column]
        C3[product_id column]
    end

    subgraph Parse [Parse Pattern]
        P1["series_id: TX_S1_P001"]
        P2["pattern: (?P<state>..)_(?P<store>..)_(?P<product>.*)"]
        P3["extracts: state=TX, store=S1, product=P001"]
    end

    subgraph Mapping [Explicit Mapping]
        M1["TX_S1 -> state: TX"]
        M2["TX_S2 -> state: TX"]
        M3["WI_S1 -> state: WI"]
    end
```

**Usage:**

```python
from xeries.hierarchy import HierarchyDefinition

# Column-based (most common)
hierarchy = HierarchyDefinition(
    levels=['state', 'store', 'product'],
    columns=['state_id', 'store_id', 'product_id']
)

# Parse pattern from series_id
hierarchy = HierarchyDefinition(
    levels=['state', 'store'],
    parse_pattern=r'(?P<state>\w{2})_(?P<store>\w+)',
    series_col='series_id'
)

# Explicit mapping
hierarchy = HierarchyDefinition(
    levels=['region'],
    mapping={
        'TX_S1': {'region': 'South'},
        'TX_S2': {'region': 'South'},
        'WI_S1': {'region': 'North'},
    }
)
```

---

### HierarchicalAggregator (`aggregator.py`)

Aggregates feature importance results across hierarchy levels.

**Aggregation Formula (SHAP):**

$$\phi_i(C_k) = \frac{1}{|C_k|} \sum_{x \in C_k} |\phi_i(x)|$$

Where:
- $\phi_i(C_k)$ = Mean absolute SHAP for feature $i$ in cohort $C_k$
- $|C_k|$ = Number of samples in cohort

```mermaid
flowchart LR
    subgraph Input
        SHAP[SHAPResult<br/>n_samples x n_features]
        Data[X DataFrame<br/>with hierarchy cols]
    end

    subgraph Aggregation
        Global["Global<br/>mean |SHAP| all data"]
        State["State Level<br/>mean |SHAP| per state"]
        Store["Store Level<br/>mean |SHAP| per store"]
    end

    subgraph Output
        Result[HierarchicalResult]
    end

    SHAP --> Global
    SHAP --> State
    SHAP --> Store
    Data --> State
    Data --> Store
    Global --> Result
    State --> Result
    Store --> Result
```

**Usage:**

```python
from xeries.hierarchy import HierarchyDefinition, HierarchicalAggregator

hierarchy = HierarchyDefinition(
    levels=['state', 'store'],
    columns=['state_id', 'store_id']
)
aggregator = HierarchicalAggregator(hierarchy)

# Aggregate SHAP results
hier_result = aggregator.aggregate_shap(
    shap_result, 
    X, 
    include_raw=True  # Store raw values for violin plots
)

# Aggregate permutation importance (re-computes per cohort)
hier_result = aggregator.aggregate_importance(
    pfi_result, X, y,
    model=model,
    metric='mse',
    n_repeats=5
)
```

---

### HierarchicalExplainer (`explainer.py`)

Wrapper that combines a base explainer with hierarchical aggregation.

```mermaid
sequenceDiagram
    participant User
    participant HierarchicalExplainer
    participant BaseExplainer
    participant HierarchicalAggregator
    participant HierarchicalResult

    User->>HierarchicalExplainer: explain(X, y)
    HierarchicalExplainer->>BaseExplainer: explain(X)
    BaseExplainer-->>HierarchicalExplainer: SHAPResult / FIResult
    HierarchicalExplainer->>HierarchicalAggregator: aggregate(result, X)
    HierarchicalAggregator-->>HierarchicalExplainer: HierarchicalResult
    HierarchicalExplainer-->>User: HierarchicalResult
```

**Usage:**

```python
from xeries import ConditionalSHAP
from xeries.hierarchy import HierarchyDefinition, HierarchicalExplainer

# Define hierarchy
hierarchy = HierarchyDefinition(
    levels=['state', 'store'],
    columns=['state_id', 'store_id']
)

# Create hierarchical explainer
base = ConditionalSHAP(model, X_train, series_col='level')
explainer = HierarchicalExplainer(base, hierarchy)

# Compute hierarchical results
result = explainer.explain(X_test, include_raw=True)

# Access at different levels
global_imp = result.get_global()
state_df = result.get_level_df('state')
store_df = result.get_level_df('store')

# Compare cohorts
comparison = explainer.compare_cohorts(X_test, level='state', top_n=5)

# Feature ranking stability
stability = explainer.feature_ranking_stability(X_test, level='store')
```

---

### HierarchicalResult (`types.py`)

Container for multi-level aggregated importance results.

**Structure:**

```mermaid
graph TB
    subgraph HierarchicalResult
        levels["levels: [global, state, store]"]
        features["features: [lag_1, lag_2, price, ...]"]
        
        subgraph importance_by_level
            global_imp["global: {all: {lag_1: 0.5, ...}}"]
            state_imp["state: {TX: {...}, WI: {...}}"]
            store_imp["store: {TX_S1: {...}, TX_S2: {...}, ...}"]
        end
        
        subgraph raw_values_by_level
            global_raw["global: {all: array(n, f)}"]
            state_raw["state: {TX: array(...), WI: array(...)}"]
        end
    end
```

**Methods:**

```python
# Get DataFrame for a level (cohorts as rows, features as columns)
state_df = result.get_level_df('state')

# Get global importance as dict
global_imp = result.get_global()

# Get cohorts at a level
cohorts = result.get_cohorts_at_level('store')  # ['TX_S1', 'TX_S2', ...]

# Get raw SHAP values for violin plots
raw_values = result.get_raw_values('state', 'TX')  # array(n_samples, n_features)
```

## Example Workflow

```mermaid
flowchart TB
    subgraph Setup
        Data[Load Data]
        Model[Train Model]
        Hier[Define Hierarchy]
    end

    subgraph Explain
        Base[Create Base Explainer<br/>ConditionalSHAP]
        HExpl[Create HierarchicalExplainer]
        Compute[Compute Explanations]
    end

    subgraph Analyze
        Global[Global Importance]
        ByLevel[Importance by Level]
        Compare[Compare Cohorts]
        Viz[Visualize]
    end

    Data --> Model
    Data --> Hier
    Model --> Base
    Hier --> HExpl
    Base --> HExpl
    HExpl --> Compute
    Compute --> Global
    Compute --> ByLevel
    Compute --> Compare
    ByLevel --> Viz
```

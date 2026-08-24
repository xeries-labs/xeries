# Partitioners Module

The partitioners module provides strategies for dividing data into subgroups for conditional permutation.

## Location

`xeries/partitioners/`

## Architecture

```mermaid
classDiagram
    class BasePartitioner {
        <<abstract>>
        +fit(X, feature) BasePartitioner
        +get_groups(X) NDArray
        +fit_get_groups(X, feature) NDArray
    }

    class TreePartitioner {
        +max_depth: int
        +min_samples_leaf: int
        +random_state: int
        +tree_: DecisionTreeRegressor
        +fit(X, feature) TreePartitioner
        +get_groups(X) NDArray
    }

    class ManualPartitioner {
        +groups: NDArray
        +fit(X, feature) ManualPartitioner
        +get_groups(X) NDArray
    }

    BasePartitioner <|-- TreePartitioner
    BasePartitioner <|-- ManualPartitioner
```

## Purpose

```mermaid
flowchart LR
    subgraph Problem [Standard Permutation]
        A1[Product A lag_1]
        B1[Product B lag_1]
        A2[Product A target]
        B2[Product B target]
        A1 -.->|"shuffled to"| B2
        B1 -.->|"shuffled to"| A2
    end

    subgraph Solution [Conditional Permutation]
        A3[Product A lag_1]
        A4[Product A lag_1']
        B3[Product B lag_1]
        B4[Product B lag_1']
        A3 -->|"shuffled within A"| A4
        B3 -->|"shuffled within B"| B4
    end
```

## Components

### TreePartitioner (`tree.py`)

Automatically learns homogeneous subgroups using a decision tree (cs-PFI method).

```mermaid
flowchart TB
    subgraph Input
        X[Feature Matrix X]
        F[Target Feature f]
    end

    subgraph TreeLearning
        Tree[DecisionTree<br/>predict feature f<br/>from other features]
        Leaves[Leaf Assignments]
    end

    subgraph Output
        Groups[Group Labels<br/>one per sample]
    end

    X --> Tree
    F --> Tree
    Tree --> Leaves
    Leaves --> Groups
```

**How it works:**
1. Train a decision tree to predict feature `f` from other features
2. Each leaf node becomes a group
3. Samples in the same leaf have similar feature relationships
4. Permuting within leaves preserves correlations

**Usage:**

```python
from xeries.partitioners import TreePartitioner

partitioner = TreePartitioner(
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)

# Fit and get groups
groups = partitioner.fit_get_groups(X, feature='lag_1')

# Use with ConditionalPermutationImportance
from xeries import ConditionalPermutationImportance

explainer = ConditionalPermutationImportance(
    model=model,
    metric='mse',
    strategy='auto',  # Uses TreePartitioner internally
    # Or provide custom partitioner:
    # partitioner=TreePartitioner(max_depth=3)
)
```

---

### ManualPartitioner (`manual.py`)

Uses pre-defined groups based on domain knowledge.

```mermaid
flowchart LR
    subgraph DomainKnowledge
        Series[Series IDs]
        Region[Region Labels]
        Custom[Custom Groups]
    end

    subgraph ManualPartitioner
        Groups[Group Array]
    end

    subgraph Usage
        PFI[ConditionalPermutation<br/>Importance]
    end

    Series --> Groups
    Region --> Groups
    Custom --> Groups
    Groups --> PFI
```

**Usage:**

```python
from xeries.partitioners import ManualPartitioner
import numpy as np

# Create groups from series IDs
series_ids = X.index.get_level_values('level')
groups = pd.factorize(series_ids)[0]

partitioner = ManualPartitioner(groups=groups)

# Use with explainer
from xeries import ConditionalPermutationImportance

explainer = ConditionalPermutationImportance(
    model=model,
    metric='mse',
    strategy='manual',
    partitioner=partitioner
)

result = explainer.explain(X, y)

# Or pass groups directly
result = explainer.explain(X, y, groups=groups)
```

## Comparison

```mermaid
flowchart TB
    subgraph TreePartitioner
        T1[Automatic]
        T2[Learns from data]
        T3[Feature-specific groups]
        T4[Good for unknown correlations]
    end

    subgraph ManualPartitioner
        M1[Manual]
        M2[Domain knowledge]
        M3[Fixed groups]
        M4[Good for known structure]
    end
```

| Feature | TreePartitioner | ManualPartitioner |
|---------|-----------------|-------------------|
| Group Definition | Automatic (learned) | Manual (user-defined) |
| Feature-Specific | Yes (different groups per feature) | No (same groups for all) |
| Requires Domain Knowledge | No | Yes |
| Computational Cost | Higher (fits tree per feature) | Lower |
| Best For | Unknown correlations | Known series structure |

## Example: Series-Based Grouping

```python
# For multi-series data, often the simplest approach is
# to permute within each series

from xeries import ConditionalPermutationImportance

# Get series IDs from MultiIndex
series_ids = X.index.get_level_values('level')
groups = pd.factorize(series_ids)[0]

explainer = ConditionalPermutationImportance(
    model=model,
    metric='mse',
    strategy='manual'
)

result = explainer.explain(X, y, groups=groups)
```

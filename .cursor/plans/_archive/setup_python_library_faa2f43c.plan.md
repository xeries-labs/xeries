---
name: Setup Python Library
overview: Create a modern Python library structure for conditional feature importance in multi-time series forecasting, using UV for package management, pytest for testing, ruff/mypy for code quality, MkDocs for documentation, and GitHub Actions for CI/CD.
todos:
  - id: init-uv
    content: Initialize UV project with pyproject.toml and core dependencies
    status: completed
  - id: package-structure
    content: Create src/tcpfi/ package structure with __init__.py files
    status: completed
  - id: core-types
    content: Create core/base.py and core/types.py with abstract classes and type definitions
    status: completed
  - id: manual-partitioner
    content: Implement partitioners/manual.py for dictionary-based grouping
    status: completed
  - id: tree-partitioner
    content: Implement partitioners/tree.py for tree-based cs-PFI
    status: completed
  - id: permutation-importance
    content: Implement importance/permutation.py (ConditionalPermutationImportance)
    status: completed
  - id: conditional-shap
    content: Implement importance/shap.py (ConditionalSHAP)
    status: completed
  - id: skforecast-adapter
    content: Implement adapters/skforecast.py for skforecast integration
    status: completed
  - id: visualization
    content: Implement visualization/plots.py for importance plots
    status: completed
  - id: test-fixtures
    content: Create tests/conftest.py with shared fixtures and sample data
    status: completed
  - id: unit-tests
    content: Write unit tests for partitioners and importance calculators
    status: completed
  - id: code-quality
    content: Configure ruff, mypy, and pre-commit
    status: completed
  - id: ci-workflow
    content: Create .github/workflows/ci.yml for testing and linting
    status: completed
  - id: release-workflow
    content: Create .github/workflows/release.yml for PyPI publishing
    status: completed
  - id: docs-setup
    content: Configure MkDocs with material theme and mkdocstrings
    status: completed
  - id: readme-license
    content: Create README.md with usage examples and LICENSE file
    status: completed
isProject: false
---

# Time-Conditional Feature Importance Library Setup

## Package Name and Structure

Based on your proposed API (`import conditional_feature_importance as cfi`), I recommend the package name `tcpfi` (Time-Conditional Permutation Feature Importance) for brevity on PyPI while maintaining the import alias.

```
time_conditional_pfi/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Test, lint, type-check on PR/push
│       ├── release.yml         # Publish to PyPI on tag
│       └── docs.yml            # Build and deploy docs
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── api/
│   │   └── reference.md
│   └── tutorials/
│       └── quickstart.md
├── src/
│   └── tcpfi/
│       ├── __init__.py
│       ├── _version.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract base classes
│       │   └── types.py        # Type definitions
│       ├── importance/
│       │   ├── __init__.py
│       │   ├── permutation.py  # ConditionalPermutationImportance
│       │   └── shap.py         # ConditionalSHAP
│       ├── partitioners/
│       │   ├── __init__.py
│       │   ├── manual.py       # Dictionary-based grouping
│       │   └── tree.py         # Tree-based cs-PFI
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── skforecast.py   # skforecast integration
│       │   └── base.py         # Adapter interface
│       └── visualization/
│           ├── __init__.py
│           └── plots.py        # Plotting utilities
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── unit/
│   │   ├── test_permutation.py
│   │   ├── test_partitioners.py
│   │   └── test_shap.py
│   └── integration/
│       └── test_skforecast.py
├── pyproject.toml              # UV + build configuration
├── uv.lock                     # UV lockfile (auto-generated)
├── .pre-commit-config.yaml
├── .gitignore
├── LICENSE
├── README.md
└── mkdocs.yml
```

## UV Package Management

The `pyproject.toml` will use UV-native configuration:

```toml
[project]
name = "tcpfi"
version = "0.1.0"
description = "Conditional Feature Importance for Multi-Time Series Forecasting"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "Your Name", email = "you@example.com" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "shap>=0.44",
    "joblib>=1.3",
]

[project.optional-dependencies]
skforecast = ["skforecast>=0.13"]
all = ["tcpfi[skforecast]"]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "pandas-stubs",
]
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tcpfi"]
```

## Code Quality Tools

### Ruff (Linting + Formatting)

```toml
# In pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.ruff.lint.isort]
known-first-party = ["tcpfi"]
```

### Mypy (Type Checking)

```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
plugins = ["numpy.typing.mypy_plugin"]
```

### Pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pandas-stubs, numpy]
```

## Test Harness (pytest)

```toml
# In pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=tcpfi --cov-report=term-missing --cov-report=xml"
markers = [
    "slow: marks tests as slow",
    "integration: integration tests requiring external libs",
]

[tool.coverage.run]
source = ["src/tcpfi"]
branch = true
```

## Documentation (MkDocs + Material)

```yaml
# mkdocs.yml
site_name: tcpfi
site_description: Conditional Feature Importance for Multi-Time Series Forecasting
repo_url: https://github.com/yourusername/tcpfi

theme:
  name: material
  features:
    - navigation.sections
    - content.code.copy
  palette:
    - scheme: default
      primary: indigo

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Tutorials:
      - Quickstart: tutorials/quickstart.md
  - API Reference: api/reference.md
```

## CI/CD with GitHub Actions

### CI Workflow (`.github/workflows/ci.yml`)

- Triggers on push/PR to main
- Matrix test: Python 3.10, 3.11, 3.12 on ubuntu-latest
- Steps: UV install, lint (ruff), type-check (mypy), test (pytest)

### Release Workflow (`.github/workflows/release.yml`)

- Triggers on version tags (v*)
- Builds wheel using `uv build`
- Publishes to PyPI using trusted publishing (OIDC)

### Docs Workflow (`.github/workflows/docs.yml`)

- Triggers on push to main
- Builds MkDocs and deploys to GitHub Pages

## Key Dependencies


| Purpose                | Package                               |
| ---------------------- | ------------------------------------- |
| Core computation       | numpy, pandas, scikit-learn           |
| SHAP values            | shap                                  |
| Parallelization        | joblib                                |
| Forecaster integration | skforecast (optional)                 |
| Testing                | pytest, pytest-cov                    |
| Linting/Formatting     | ruff                                  |
| Type checking          | mypy                                  |
| Documentation          | mkdocs, mkdocs-material, mkdocstrings |



"""MkDocs hooks: copy example notebooks into docs/ for mkdocs-jupyter."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_EXAMPLES = REPO_ROOT / "examples"
DST_EXAMPLES = REPO_ROOT / "docs" / "examples"


def on_pre_build(config) -> None:  # noqa: ARG001
    """Mirror ``examples/*.ipynb`` under ``docs/examples`` before the build."""
    if DST_EXAMPLES.exists():
        shutil.rmtree(DST_EXAMPLES)
    shutil.copytree(
        SRC_EXAMPLES,
        DST_EXAMPLES,
        ignore=shutil.ignore_patterns("README.md", ".ipynb_checkpoints"),
    )

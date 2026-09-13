"""Tests for example notebooks."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def get_notebook_files() -> list[Path]:
    """Return all notebooks under examples/, excluding checkpoints."""
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(
        path
        for path in EXAMPLES_DIR.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )


def _rel(path: Path) -> str:
    return path.relative_to(EXAMPLES_DIR).as_posix()


NOTEBOOKS = get_notebook_files()
NOTEBOOK_IDS = [_rel(path) for path in NOTEBOOKS]


@pytest.fixture
def notebook_executor():
    """Fixture to execute notebooks."""

    def _execute(notebook_path: Path, timeout: int = 300) -> tuple[bool, str]:
        """Execute a notebook and return success status and output."""
        try:
            with notebook_path.open(encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            client = NotebookClient(
                notebook,
                timeout=timeout,
                allow_errors=False,
                kernel_name="python3",
            )
            client.execute()
            return True, ""
        except Exception:
            return False, traceback.format_exc()

    return _execute


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebook_executes(notebook_path: Path, notebook_executor) -> None:
    """Execute each example notebook without errors."""
    success, output = notebook_executor(notebook_path)
    if not success:
        pytest.fail(f"Notebook execution failed ({_rel(notebook_path)}):\n{output}")


@pytest.mark.notebook
def test_notebooks_are_discovered() -> None:
    """Nested example notebooks must be found (not only the examples/ root)."""
    assert NOTEBOOKS, "No notebooks found under examples/"
    nested = [path for path in NOTEBOOKS if path.parent != EXAMPLES_DIR]
    assert nested, "Expected notebooks in examples/ subdirectories"
    assert any("quickstart" in _rel(path) for path in NOTEBOOKS)


@pytest.mark.notebook
def test_notebooks_are_valid_json() -> None:
    """Test that all notebooks are valid JSON."""
    notebooks = get_notebook_files()
    assert notebooks, "No notebooks found"

    for nb_path in notebooks:
        try:
            with nb_path.open(encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {_rel(nb_path)}: {e}")


@pytest.mark.notebook
def test_notebooks_have_valid_structure() -> None:
    """Test that all notebooks have valid nbformat structure."""
    notebooks = get_notebook_files()
    assert notebooks, "No notebooks found"

    for nb_path in notebooks:
        with nb_path.open(encoding="utf-8") as f:
            nb = json.load(f)

        assert "cells" in nb, f"Missing 'cells' in {_rel(nb_path)}"
        assert "metadata" in nb, f"Missing 'metadata' in {_rel(nb_path)}"
        assert "nbformat" in nb, f"Missing 'nbformat' in {_rel(nb_path)}"
        assert nb["nbformat"] >= 4, f"nbformat too old in {_rel(nb_path)}"

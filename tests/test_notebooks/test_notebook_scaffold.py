from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

EXPECTED_NOTEBOOKS = [
    "attack_experiment_synthesis.ipynb",
]


def test_all_notebooks_exist_and_start_with_attack_tests(repo_root: Path) -> None:
    notebook_dir = repo_root / "notebooks"
    assert sorted(path.name for path in notebook_dir.glob("*.ipynb")) == EXPECTED_NOTEBOOKS
    # The pytest-runner cell must appear in one of the first two code cells:
    # slot 0 when the notebook has no chdir-bootstrap, slot 1 when
    # `scripts/add_chdir_cell.py` has prepended one.
    for name in EXPECTED_NOTEBOOKS:
        notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
        leading_code = [c for c in notebook["cells"][:2] if c["cell_type"] == "code"]
        assert leading_code, f"{name}: no leading code cell"
        sources = ["".join(c["source"]) for c in leading_code]
        assert any("pytest tests/test_attacks/ -q" in s for s in sources), (
            f"{name}: pytest runner not found in first two code cells"
        )


def test_notebook_bootstrap_uses_pyproject_toml_not_checkpoints(repo_root: Path) -> None:
    """Regression: bootstrap must not use checkpoints/ which is gitignored."""
    notebook_dir = repo_root / "notebooks"
    for name in EXPECTED_NOTEBOOKS:
        notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
        first_code = next(c for c in notebook["cells"] if c["cell_type"] == "code")
        source = "".join(first_code["source"])
        assert "checkpoints" not in source, (
            f"{name}: bootstrap still references 'checkpoints' (gitignored directory)"
        )
        assert "pyproject.toml" in source, (
            f"{name}: bootstrap does not use 'pyproject.toml' as root marker"
        )


def test_src_reporting_imports_in_notebooks_resolve(repo_root: Path) -> None:
    pattern = re.compile(r"from (src\.reporting(?:\.\w+)*) import ")
    for name in EXPECTED_NOTEBOOKS:
        notebook = json.loads((repo_root / "notebooks" / name).read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            for module_name in pattern.findall(source):
                assert importlib.util.find_spec(module_name) is not None, (
                    f"{name}: {module_name} does not resolve"
                )

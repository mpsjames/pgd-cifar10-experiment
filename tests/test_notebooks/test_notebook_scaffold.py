from __future__ import annotations

import json
from pathlib import Path

EXPECTED_NOTEBOOKS = [
    "01_research_protocol.ipynb",
    "02_baseline_clean_models.ipynb",
    "03_attack_implementation_validation.ipynb",
    "04_main_quantitative_results.ipynb",
    "05_vulnerability_analysis.ipynb",
    "06_qualitative_visualization.ipynb",
    "07a_adv_training_resnet18.ipynb",
    "07b_adv_training_wrn34_10.ipynb",
    "07c_adv_training_vit_tiny.ipynb",
    "08_defense_evaluation_synthesis.ipynb",
    "09_transfer_attack_analysis.ipynb",
    "10_architecture_robustness_comparison.ipynb",
    "11_discussion_and_limitations.ipynb",
]


def test_all_notebooks_exist_and_start_with_attack_tests(repo_root: Path) -> None:
    notebook_dir = repo_root / "notebooks"
    assert sorted(path.name for path in notebook_dir.glob("*.ipynb")) == EXPECTED_NOTEBOOKS
    for name in EXPECTED_NOTEBOOKS:
        notebook = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
        first_cell = notebook["cells"][0]
        assert first_cell["cell_type"] == "code"
        assert "pytest tests/test_attacks/ -q" in "".join(first_cell["source"])


def test_wrn_and_discussion_include_fallback_disclosure(repo_root: Path) -> None:
    wrn = (repo_root / "notebooks/07b_adv_training_wrn34_10.ipynb").read_text(encoding="utf-8")
    discussion = (repo_root / "notebooks/11_discussion_and_limitations.ipynb").read_text(
        encoding="utf-8"
    )
    assert "RobustBench" in wrn
    assert "fallback" in wrn
    assert "WRN RobustBench fallback" in discussion

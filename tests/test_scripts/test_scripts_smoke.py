from __future__ import annotations

import importlib


def test_script_modules_import() -> None:
    for module in [
        "scripts.train_clean",
        "scripts.train_adversarial",
        "scripts.run_white_box",
        "scripts.run_transfer",
        "scripts.run_epsilon_sweep",
        "scripts.download_robustbench_wrn",
    ]:
        importlib.import_module(module)

"""Acceptance gates used by notebook summary tables.

The thresholds are project-level reporting gates from the experiment plan; they
are not model-selection logic.
"""

CLEAN_ACC_GATES = {
    "resnet18": 0.93,
    "wrn_34_10": 0.95,
    "vit_tiny": 0.85,
}

AT_GATES = {
    "resnet18": (0.80, 0.42),
    "wrn_34_10": (0.83, 0.48),
    "vit_tiny": (0.75, 0.40),
}

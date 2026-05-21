"""Acceptance gates used by notebook summary tables.

The thresholds are project-level reporting gates from the experiment plan; they
are not model-selection logic.
"""

CLEAN_ACC_GATES = {
    "resnet18": 0.93,
    "vit_tiny": 0.85,
}

AT_GATES = {
    "resnet18": (0.80, 0.42),
    "vit_tiny": (0.75, 0.40),
}

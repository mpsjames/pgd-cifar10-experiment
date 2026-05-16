"""Load external RobustBench checkpoints used as the WRN fallback path."""

from __future__ import annotations

from torch import nn


def load_robustbench_wrn(
    model_name: str = "Madry2018", dataset: str = "cifar10", threat_model: str = "Linf"
) -> nn.Module:
    """Load the WRN fallback checkpoint from RobustBench.

    Args:
        model_name: RobustBench model identifier.
        dataset: Dataset identifier understood by RobustBench.
        threat_model: Threat-model identifier understood by RobustBench.

    Returns:
        Loaded `nn.Module` ready for evaluation or checkpoint export.

    Raises:
        RuntimeError: When the optional `robustbench` dependency is not
            installed.
    """
    try:
        from robustbench.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "RobustBench fallback requested but robustbench is not installed"
        ) from exc
    return load_model(model_name=model_name, dataset=dataset, threat_model=threat_model)

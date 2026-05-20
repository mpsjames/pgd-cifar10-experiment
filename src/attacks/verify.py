"""Fail fast when adversarial samples violate documented attack constraints."""

from __future__ import annotations

from torch import Tensor


def verify_perturbation(x_orig: Tensor, x_adv: Tensor, epsilon: float, norm: str = "Linf") -> None:
    """Verify perturbation norm and pixel-domain constraints.

    Args:
        x_orig: Clean inputs, shape `(B, C, H, W)`, dtype float-compatible.
        x_adv: Adversarial inputs, same shape and dtype as `x_orig`.
        epsilon: Maximum allowed perturbation magnitude in normalized image
            space.
        norm: Constraint set to verify. Only `"Linf"` and `"L2"` are
            supported.

    Raises:
        AssertionError: When shapes differ, the perturbation exceeds
            `epsilon`, or `x_adv` leaves the `[0, 1]` image domain.
        ValueError: When `norm` is unsupported.
    """
    if x_orig.shape != x_adv.shape:
        raise AssertionError(f"Shape mismatch: {tuple(x_orig.shape)} != {tuple(x_adv.shape)}")
    delta = (x_adv - x_orig).abs()

    if norm == "Linf":
        max_perturb = delta.max().item()
        if max_perturb > epsilon + 1e-6:
            raise AssertionError(f"L_inf violation: {max_perturb:.6f} > {epsilon:.6f}")
    elif norm == "L2":
        l2_perturb = delta.flatten(1).norm(p=2, dim=1).max().item()
        if l2_perturb > epsilon + 1e-6:
            raise AssertionError(f"L2 violation: {l2_perturb:.6f} > {epsilon:.6f}")
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    if x_adv.min().item() < -1e-6 or x_adv.max().item() > 1.0 + 1e-6:
        raise AssertionError("Adversarial examples are outside the valid image domain [0, 1]")

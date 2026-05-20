"""Implement the Linf Square Attack black-box adversary."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.attacks.base import BaseAttack
from src.attacks.square_core import bernoulli_sign, margin_loss, schedule_p


class SquareAttack(BaseAttack):
    """Apply query-only Linf random search to raw `[0, 1]` images."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if config.p_init is None or config.loss is None or config.seed is None:
            raise ValueError("SquareAttack requires p_init, loss, and seed on AttackConfig")
        if not 0.0 < config.p_init <= 1.0:
            raise ValueError(f"p_init must be in (0, 1], got {config.p_init}")
        self.last_query_count: int = 0

    def perturb(self, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
        """Generate adversarial examples via Linf random search."""
        was_training, x_orig = self._prepare(model, x, y)
        if self.config.epsilon <= 0.0 or self.config.num_steps == 0:
            self.last_query_count = 0
            self._restore(model, was_training)
            return x.clone()

        try:
            device = x.device
            generator = torch.Generator(device=device).manual_seed(int(self.config.seed))
            b, c, h, w = x_orig.shape
            epsilon = float(self.config.epsilon)
            n_features = c * h * w

            stripe_signs = bernoulli_sign((b, c, 1, w), generator, device)
            x_adv = self._project_linf(
                x_orig, x_orig + stripe_signs.expand(b, c, h, w) * epsilon, epsilon
            )
            with torch.no_grad():
                loss = self._score_loss(model(x_adv), y)

            query_count = 1
            for step in range(1, int(self.config.num_steps)):
                side = self._square_side(step, n_features, c, h, w)
                row = torch.randint(0, h - side + 1, (b,), generator=generator, device=device)
                col = torch.randint(0, w - side + 1, (b,), generator=generator, device=device)
                sign = bernoulli_sign((b, c, 1, 1), generator, device) * epsilon
                candidate = self._patch_candidate(x_orig, x_adv, row, col, side, sign, epsilon)

                with torch.no_grad():
                    cand_loss = self._score_loss(model(candidate), y)
                query_count += 1
                improved = cand_loss < loss
                if improved.any():
                    idx = improved.nonzero(as_tuple=False).squeeze(1)
                    x_adv[idx] = candidate[idx]
                    loss[idx] = cand_loss[idx]

            self.last_query_count = query_count
            return x_adv.detach().to(dtype=x.dtype)
        finally:
            self._restore(model, was_training)

    def _score_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        if self.config.loss == "margin":
            return margin_loss(logits, y)
        return -F.cross_entropy(logits, y, reduction="none")

    def _square_side(self, step: int, n_features: int, channels: int, h: int, w: int) -> int:
        p = schedule_p(step, int(self.config.num_steps), float(self.config.p_init))
        return min(max(int(round((p * n_features / channels) ** 0.5)), 1), h, w)

    def _patch_candidate(
        self,
        x_orig: Tensor,
        x_adv: Tensor,
        row: Tensor,
        col: Tensor,
        side: int,
        sign: Tensor,
        epsilon: float,
    ) -> Tensor:
        candidate = x_adv.clone()
        for sample_idx in range(x_orig.size(0)):
            r0 = int(row[sample_idx].item())
            c0 = int(col[sample_idx].item())
            patch = x_orig[sample_idx, :, r0 : r0 + side, c0 : c0 + side]
            candidate[sample_idx, :, r0 : r0 + side, c0 : c0 + side] = torch.clamp(
                patch + sign[sample_idx], 0.0, 1.0
            )
        return self._project_linf(x_orig, candidate, epsilon)

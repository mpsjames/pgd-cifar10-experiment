"""Early-stopping helper shared by training service objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track a maximized validation metric and signal when to stop.

    Patience counts validation events (not epochs); with
    `val_every_n_epochs > 1`, one tick spans multiple training epochs.
    Disabled when `patience <= 0`; in that mode `update` is a no-op.
    """

    patience: int
    min_delta: float = 0.0
    best: float = float("-inf")
    stale_ticks: int = 0
    should_stop: bool = False

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def update(self, score: float) -> bool:
        """Record one validation score; return True iff training should stop."""
        if not self.enabled:
            return False
        if score > self.best + self.min_delta:
            self.best = score
            self.stale_ticks = 0
            self.should_stop = False
        else:
            self.stale_ticks += 1
            if self.stale_ticks >= self.patience:
                self.should_stop = True
        return self.should_stop

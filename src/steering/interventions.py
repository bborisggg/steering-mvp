"""Steering interventions: B0, B1, B2 -- Step 1's gate needs these three and nothing further.

Named after PLAN.md's own convention, not the exploratory repo's. PLAN.md defines
``r = ||alpha*v|| / E||h||`` and sweeps ``r`` (its alpha grid is a grid of *this*, e.g.
``0, 0.1, ..., 3.0``). The exploratory repo's constructors took a parameter it called ``alpha``
that meant this same relative quantity, which reads as the *absolute* push to anyone coming from
PLAN.md's own notation. Constructors here take ``r`` and a ``scale`` (``E||h||``, from
``spaces.activation_scale``) and compute the absolute push internally as ``alpha = r * scale``.

**Every intervention skips the attention sink** (DECISIONS D4, decided 2026-08-22): position 0 is
touched by nothing, steering included. The mechanism is exactly the one proven in
``hooks.py``'s prefill-only test -- under a KV cache, ``seq_len > 1`` identifies precisely the
prefill pass, so "skip index 0 when seq_len > 1" skips exactly the sink with no separate position
bookkeeping to keep in sync with the cache.
"""

from __future__ import annotations

import torch


class Intervention:
    """Base class. Subclasses implement :meth:`apply` on the non-sink positions only."""

    id: str = "B0"

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            head, tail = hidden[:, :1, :], hidden[:, 1:, :]
            return torch.cat([head, self.apply(tail)], dim=1)
        return self.apply(hidden)


class NoSteering(Intervention):
    """B0: the unsteered reference. Touches nothing, including the sink."""

    id = "B0"

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class AdditiveSteering(Intervention):
    """B1: ``h + alpha*v`` -- the assignment's baseline.

    Args:
        v: any nonzero vector; normalised to unit length internally, so ``alpha`` alone carries
            the scale.
        r: relative strength, PLAN.md's ``r = ||alpha*v|| / scale``.
        scale: ``E||h||`` for the prompts being steered (``spaces.activation_scale``), *not* a
            corpus-wide value -- DECISIONS D2.
    """

    id = "B1"

    def __init__(self, v: torch.Tensor, r: float, scale: float):
        self.v = v / v.norm()
        self.r = r
        self.scale = scale

    @property
    def alpha(self) -> float:
        return self.r * self.scale

    @property
    def delta(self) -> torch.Tensor:
        return self.alpha * self.v

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.delta


class NormMatchedSteering(AdditiveSteering):
    """B2: steer, then rescale each token back to its original norm. The mandatory control.

    Concept lives mostly in direction; restoring the norm recovers much of the fluency a plain
    push costs. A denoiser that does not beat this has not demonstrated anything.
    """

    id = "B2"

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        original = hidden.norm(dim=-1, keepdim=True)
        steered = hidden + self.delta
        return steered * (original / steered.norm(dim=-1, keepdim=True).clamp_min(1e-6))

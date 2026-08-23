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
        # .to(hidden.dtype): self.delta is built from v, which callers construct in whatever
        # dtype is convenient (float32, typically) -- a no-op for GPT-2, whose hidden states
        # already are float32, but required for Gemma (bf16): ResidualHook refuses to let an
        # intervention silently change the hook's dtype (hooks.py's own guard), and it is right
        # to -- Gemma's later blocks expect bf16 back, not a promoted float32 residual stream.
        return hidden + self.delta.to(hidden.dtype)


class NormMatchedSteering(AdditiveSteering):
    """B2: steer, then rescale each token back to its original norm. The mandatory control.

    Concept lives mostly in direction; restoring the norm recovers much of the fluency a plain
    push costs. A denoiser that does not beat this has not demonstrated anything.
    """

    id = "B2"

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        original = hidden.norm(dim=-1, keepdim=True)
        steered = super().apply(hidden)  # dtype-safe addition, not a duplicated raw one
        return steered * (original / steered.norm(dim=-1, keepdim=True).clamp_min(1e-6))


class DenoisedSteering(AdditiveSteering):
    """``D(h + alpha*v)`` -- steer, then denoise. Step 8's headline arm: the whole point of
    this project's denoiser.

    ``t`` is derived from ``r`` via the corruption family's own inverse map
    (``corruptions.t_for_r``), not assumed -- DECISIONS D3 names passing a raw or fixed ``t``
    regardless of the deployed ``r`` as a silent, severe misuse: a t-conditioned denoiser
    defaults to maximum denoising at every strength if the map is skipped.

    ``project_correction=True`` turns this into PDS (Step 10): the denoiser's correction is
    applied only in the directions orthogonal to ``v``, leaving the component of the steered
    activation along ``v`` untouched --

        delta = D(steered) - steered
        delta_perp = delta - <delta, v_hat> * v_hat
        output = steered + delta_perp

    A cheap, one-shot diagnostic (PLAN.md Step 10: "no lambda sweep, no attempt to rescue it"),
    not a tuned method -- no correction-strength knob. Reusing this class rather than a
    separate PDS pipeline means it gets exactly the same alpha normalisation, r->t
    conditioning, and sink handling as every other arm for free; a hand-rolled parallel
    implementation could drift from any of the three without anyone noticing.

    Args:
        v, r, scale: as ``AdditiveSteering`` -- the steering push applied before denoising.
        model: a trained (or, for testing, untrained) ``ResidualMLPDenoiser``.
        corruption_description: the training corruption's ``describe()`` output, exactly what
            ``t_for_r`` needs and no more -- this class never needs the SAE or the split that
            originally built the corruption, only the small dict a checkpoint carries.
        project_correction: if True, drop the component of the denoiser's correction that lies
            along ``v`` (PDS). Default False: plain ``D(h + alpha*v)``.
    """

    id = "denoised"

    def __init__(self, v: torch.Tensor, r: float, scale: float, model,
                corruption_description: dict, project_correction: bool = False):
        from steering import corruptions

        super().__init__(v, r, scale)
        self.model = model
        self.t = corruptions.t_for_r(corruption_description, r)
        self.project_correction = project_correction

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        steered = super().apply(hidden)
        denoised = self.model(steered, t=self.t)
        delta = denoised - steered
        if self.project_correction:
            # self.v is already unit-normalised, by AdditiveSteering.__init__.
            coeff = torch.einsum("...d,d->...", delta, self.v)
            delta = delta - coeff[..., None] * self.v
        return steered + delta

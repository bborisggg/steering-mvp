"""Training corruptions for the denoiser -- D1 Gaussian, D2 variance-preserving, D3 rank-1 with
a fixed pool, D4 rank-1 with the full pool (PLAN.md's method table).

**The holdout is the one property that must never be wrong**, and doubly so here: D3/D4 are the
one place a real steering direction -- a DEV or TEST decoder row -- could enter training at all.
Both draw only from :meth:`VectorSplit.train_pool`, structurally disjoint from ``dev``/``test``
by construction (vectors.py: TRAIN is the *complement*, never a separately-maintained list) --
and :class:`StructuredRank1`'s constructor still asserts disjointness explicitly, defense in
depth against a wrong subsetting implementation rather than trust in the type alone, matching
:class:`VectorSplit`'s own redundant check.

D2 is PLAN.md's own formula, ``sqrt(1-beta)*h + sqrt(beta)*eps`` -- a variance-preserving
SDE-style corruption, and a genuinely different family from the exploratory repo's own "C2" (a
plain linear interpolant, ``t*h + (1-t)*eps``) despite the superficial resemblance in name.
Implemented from PLAN.md's formula directly.

D1's sigma and D3/D4's rho are both sampled **log-uniformly over [0.05, 3]** (PLAN.md §3): "so
one model covers a range of magnitudes instead of overfitting a single noise level" (the same
reasoning ported from the exploratory repo's C1). The [0,1] conditioning signal `t` for both is
the log-uniform position within that range (DECISIONS D3: normalised per family, checkpoint
stores the family plus the inverse map so a deployed `r` can be mapped to the level the model
was actually trained at). D2's conditioning is `beta` directly -- already in [0,1], no map
needed, though the r->beta correspondence has no principled derivation (see
:meth:`VariancePreserving.t_for_r`'s docstring) and would need the same empirical validation
DECISIONS D3 already flags for D1/D3/D4's own maps before being trusted at high `r`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Corrupted:
    """One training example: corrupted input, clean target, conditioning level.

    ``direction`` is the unit vector the corruption was applied along, for the rank-1 families
    that have one -- always a *corruption* direction, never an evaluation steering vector.
    """

    x: torch.Tensor        # (batch, d_model) corrupted
    target: torch.Tensor   # (batch, d_model) clean
    t: torch.Tensor        # (batch,) conditioning level in [0, 1]
    direction: torch.Tensor | None = None  # (batch, d_model) unit, rank-1 families only


class Corruption:
    """Base class. ``id`` is used in checkpoints, configs, and the report."""

    id: str = "D0"

    def __call__(self, clean: torch.Tensor, generator: torch.Generator | None = None) -> Corrupted:
        raise NotImplementedError

    def t_for_r(self, r: float) -> float:
        """The conditioning level matching a deployed steering strength ``r``.

        This connects training to deployment: passing ``t=1`` regardless of ``r`` is a silent,
        severe misuse (DECISIONS D3).
        """
        return 1.0

    def describe(self) -> dict[str, object]:
        return {"corruption": self.id}


def _check_generator(device, generator) -> None:
    """torch requires the generator's device to match the tensor's.

    Raised rather than worked around: silently sampling on another device and copying would
    change the random stream, so two runs at the same seed would diverge for a reason nothing
    records.
    """
    if generator is not None and generator.device.type != torch.device(device).type:
        raise RuntimeError(
            f"generator is on {generator.device.type} but activations are on "
            f"{torch.device(device).type}; construct the generator with "
            f"torch.Generator(device=...) matching the data"
        )


def _rand(shape, device, dtype, generator):
    _check_generator(device, generator)
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def _randn(shape, device, dtype, generator):
    _check_generator(device, generator)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _log_uniform_t(value: float, lo: float, hi: float) -> float:
    """[0,1]-normalised position of ``value`` in a log-uniform range. Shared by D1 (sigma) and
    the rank-1 families (rho) -- PLAN.md samples both log-uniformly over the same [0.05, 3]."""
    value = min(max(value, lo), hi)
    return float((math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)))


class Gaussian(Corruption):
    """D1: ``h + sigma*eps``, sigma log-uniform over ``[sigma_min, sigma_max]``.

    Per-token scale: sigma is relative to *that token's own* RMS magnitude
    (``||h|| / sqrt(d)``), not a single global constant, so the corruption adapts to whatever a
    given token's actual scale is.
    """

    id = "D1"

    def __init__(self, sigma_min: float = 0.05, sigma_max: float = 3.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, clean: torch.Tensor, generator=None) -> Corrupted:
        b, d = clean.shape
        u = _rand((b, 1), clean.device, clean.dtype, generator)
        log_min, log_max = math.log(self.sigma_min), math.log(self.sigma_max)
        sigma = torch.exp(log_min + u * (log_max - log_min)).to(clean.dtype)

        token_scale = clean.norm(dim=-1, keepdim=True) / (d**0.5)
        noise = _randn(clean.shape, clean.device, clean.dtype, generator)
        x = clean + sigma * token_scale * noise

        t = (torch.log(sigma.squeeze(-1)) - log_min) / (log_max - log_min)
        return Corrupted(x=x, target=clean, t=t.clamp(0, 1))

    def t_for_r(self, r: float) -> float:
        return _log_uniform_t(r, self.sigma_min, self.sigma_max)

    def describe(self):
        return {**super().describe(), "sigma_min": self.sigma_min, "sigma_max": self.sigma_max}


class VariancePreserving(Corruption):
    """D2: ``sqrt(1-beta)*h + sqrt(beta)*eps``, beta ~ U[0,1] -- PLAN.md's own formula.

    beta=0 is clean, beta=1 is pure noise. Conditioning ``t`` is beta directly; it is already
    in [0,1], so no log-uniform map is needed the way D1/D3/D4 need one.
    """

    id = "D2"

    def __call__(self, clean: torch.Tensor, generator=None) -> Corrupted:
        b, d = clean.shape
        beta = _rand((b, 1), clean.device, clean.dtype, generator)
        token_scale = clean.norm(dim=-1, keepdim=True) / (d**0.5)
        noise = _randn(clean.shape, clean.device, clean.dtype, generator) * token_scale
        x = torch.sqrt(1 - beta) * clean + torch.sqrt(beta) * noise
        return Corrupted(x=x, target=clean, t=beta.squeeze(-1))

    def t_for_r(self, r: float) -> float:
        """No principled r->beta correspondence exists -- clamped to [0,1] as an honest
        placeholder, the same choice the exploratory repo made for its structurally-analogous
        C2, and flagged there and here as unvalidated (DECISIONS D3: an inherited r->t map
        turned out wrong by 2-3x on a different corruption family, and nobody had chosen it
        deliberately; this one has not been checked at all yet)."""
        return float(min(max(r, 0.0), 1.0))


class Rank1(Corruption):
    """D0 base: ``h + rho*||h||*u`` for a unit direction ``u``, rho log-uniform over
    ``[rho_min, rho_max]``. Directions are drawn uniformly from the sphere by default --
    steering-shaped but structure-free. :class:`StructuredRank1` overrides where the direction
    comes from for D3/D4.
    """

    id = "D0-rank1"

    def __init__(self, rho_min: float = 0.05, rho_max: float = 3.0):
        self.rho_min = rho_min
        self.rho_max = rho_max

    def _directions(self, b: int, d: int, device, dtype, generator) -> torch.Tensor:
        u = _randn((b, d), device, dtype, generator)
        return u / u.norm(dim=-1, keepdim=True)

    def __call__(self, clean: torch.Tensor, generator=None) -> Corrupted:
        b, d = clean.shape
        u = self._directions(b, d, clean.device, clean.dtype, generator)
        frac = _rand((b, 1), clean.device, clean.dtype, generator)
        log_min, log_max = math.log(self.rho_min), math.log(self.rho_max)
        rho = torch.exp(log_min + frac * (log_max - log_min)).to(clean.dtype)

        token_norm = clean.norm(dim=-1, keepdim=True)
        x = clean + rho * token_norm * u

        t = (torch.log(rho.squeeze(-1)) - log_min) / (log_max - log_min)
        return Corrupted(x=x, target=clean, t=t.clamp(0, 1), direction=u)

    def t_for_r(self, r: float) -> float:
        return _log_uniform_t(r, self.rho_min, self.rho_max)

    def describe(self):
        return {**super().describe(), "rho_min": self.rho_min, "rho_max": self.rho_max}


class StructuredRank1(Rank1):
    """Rank-1 corruption drawn from a pool of SAE decoder-row directions, resampled fresh each
    call -- never any DEV/TEST feature. Base for D3 and D4, which differ only in whether the
    pool is a fixed subsample (D3) or the entire train pool (D4); see those subclasses.

    ``pool_indices`` and ``holdout`` are taken as plain values here, not a ``VectorSplit`` --
    so the disjointness check below is exercised directly against whatever was actually passed,
    not against a trusted intermediate. D3/D4 build these from ``VectorSplit`` themselves.
    """

    id = "D-structured-rank1"

    def __init__(
        self,
        decoder: torch.Tensor,
        pool_indices: list[int],
        holdout: set[int],
        rho_min: float = 0.05,
        rho_max: float = 3.0,
    ):
        super().__init__(rho_min, rho_max)
        overlap = set(pool_indices) & set(holdout)
        if overlap:
            raise ValueError(
                f"HOLDOUT VIOLATION: corruption pool contains {len(overlap)} DEV/TEST "
                f"features: {sorted(overlap)[:10]}. The denoiser must never see them."
            )
        self.pool_indices = sorted(pool_indices)
        # detach: W_dec is an nn.Parameter: without this the corruption drags the SAE into the
        # autograd graph and the second training step fails on an already-freed graph.
        pool = decoder[torch.tensor(self.pool_indices, device=decoder.device)].detach()
        self.pool = pool / pool.norm(dim=-1, keepdim=True)

    def _directions(self, b: int, d: int, device, dtype, generator) -> torch.Tensor:
        gen_device = generator.device if generator is not None else "cpu"
        idx = torch.randint(0, self.pool.shape[0], (b,), device=gen_device, generator=generator)
        return self.pool[idx.to(self.pool.device)].to(device=device, dtype=dtype)

    def describe(self):
        return {**super().describe(), "pool_size": len(self.pool_indices)}


class FixedPoolRank1(StructuredRank1):
    """D3: rank-1 corruption from a FIXED pool (default 256), chosen once via ``seed`` from
    the frozen split's train pool and frozen for the rest of training.

    Memorisation risk, measured in the exploratory repo: a fixed 256-direction pool's held-out
    reconstruction advantage sat at ratio ~400 against ~1.5-6 for a resampled pool -- a fixed
    pool this size is memorised, not generalised from. Kept as a comparison arm regardless,
    since PLAN.md's own method table calls for it.
    """

    id = "D3"

    def __init__(self, decoder, split, pool_size: int = 256, seed: int = 0,
                rho_min: float = 0.05, rho_max: float = 3.0):
        candidates = split.train_pool()
        if pool_size > len(candidates):
            raise ValueError(
                f"pool_size {pool_size} exceeds the train pool ({len(candidates)} features)"
            )
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(candidates), generator=generator).tolist()
        chosen = sorted(candidates[i] for i in order[:pool_size])
        super().__init__(decoder, chosen, split.holdout, rho_min, rho_max)
        self.pool_seed = seed

    def describe(self):
        return {**super().describe(), "pool_seed": self.pool_seed}


class FullPoolRank1(StructuredRank1):
    """D4: rank-1 corruption sampled fresh, every call, from the ENTIRE train pool (the full
    SAE dictionary minus DEV/TEST) -- no fixed subset to memorise."""

    id = "D4"

    def __init__(self, decoder, split, rho_min: float = 0.05, rho_max: float = 3.0):
        super().__init__(decoder, split.train_pool(), split.holdout, rho_min, rho_max)


def t_for_r(description: dict, r: float) -> float:
    """Map a deployed steering strength to the training conditioning level, from a stored
    description alone -- rebuilds only what the mapping needs, so a checkpoint is usable
    without the SAE or the feature split a D3/D4 corruption originally required.
    """
    kind = str(description.get("corruption", "")).upper()
    if kind == "D1":
        return Gaussian(
            description.get("sigma_min", 0.05), description.get("sigma_max", 3.0)
        ).t_for_r(r)
    if kind == "D2":
        return VariancePreserving().t_for_r(r)
    if kind in ("D0-RANK1", "D3", "D4"):
        return Rank1(
            description.get("rho_min", 0.05), description.get("rho_max", 3.0)
        ).t_for_r(r)
    return 1.0

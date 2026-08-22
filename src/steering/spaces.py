"""Activation space: centering, scale, and the encode/decode pair every module shares.

Two facts drive this module, both measured on real forwards rather than assumed -- see DEVLOG
for the numbers -- and both hold regardless of which model this MVP points at:

**Centering is architecture-dependent, not a preference.** LayerNorm (GPT-2) subtracts the
per-token mean before every read of the residual stream, so the mean component along d_model
provably cannot affect any prediction; projecting it out is exactly output-neutral (measured:
3.4e-7 relative logit change). RMSNorm (Gemma-2) divides by the RMS but never subtracts a mean,
so the same operation is a real perturbation there (measured: 3.2e-2, ~1e5x larger). Centering an
RMSNorm model is not a smaller version of the same idea -- it is a different, wrong idea.

**Scale must exclude the attention sink.** Position 0 of GPT-2's residual stream carries
30-40x the median norm of every other position (DEVLOG). A scale used to define "one typical
push" -- which is what `activation_scale` is, and what the `r` steering-strength convention in
PLAN.md is built on -- has to describe the typical token, not the sink. Median rather than mean,
because even after dropping position 0 a few outlier dimensions would still skew a mean.

**What encode/decode round-trips to.** Centering is written back into the stream, not undone on
decode: for GPT-2 that is safe (output-neutral) and for Gemma centering never happens in the
first place. So ``decode(encode(h)) == center(h)``, not ``== h``. The only quantity encode/decode
promises to restore exactly is the *scale*.
"""

from __future__ import annotations

import torch

# Architectures this project actually points at, keyed by whatever a caller might name the
# model (HF repo id or short alias). Explicit and closed rather than inferred from config
# attributes: guessing wrong here means silently centering an RMSNorm model, which perturbs it.
_CENTERS = {
    "gpt2": True,             # LayerNorm -- centering is output-neutral (measured)
    "google/gemma-2-2b-it": False,  # RMSNorm -- centering perturbs the model (measured)
    "google/gemma-2-2b": False,
}


def should_center(model_name: str) -> bool:
    """Whether this architecture's residual stream may be safely centered.

    Raises rather than defaulting: an unlisted architecture must be measured (see
    ``measure_gemma_center.py``-style check in DEVLOG) before it is added here, not guessed.
    """
    if model_name not in _CENTERS:
        raise ValueError(
            f"unknown architecture {model_name!r}: whether centering is output-neutral has "
            f"not been measured for it. Measure the relative logit change under centering "
            f"(see DEVLOG 2026-08-22) and add it to spaces._CENTERS before using this model."
        )
    return _CENTERS[model_name]


def center(hidden: torch.Tensor) -> torch.Tensor:
    """Project out the all-ones direction: subtract the per-token mean along d_model."""
    return hidden - hidden.mean(dim=-1, keepdim=True)


# encode/decode below take a `center` *keyword*, which shadows this function inside their
# bodies -- captured here under a private name rather than reached for via globals().
_center = center


def token_norms(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    exclude_sink: bool = True,
) -> torch.Tensor:
    """Per-token residual norms, flattened, with padding and (optionally) position 0 dropped.

    Args:
        hidden: ``[batch, seq, d_model]``.
        attention_mask: ``[batch, seq]``, 1 for real tokens. Required whenever the batch is
            padded, or padded positions (often exactly zero, sometimes garbage) enter the
            statistic.
        exclude_sink: drop position 0 of every sequence -- the attention sink.
    """
    start = 1 if exclude_sink else 0
    hidden = hidden[:, start:, :]
    mask = attention_mask[:, start:].bool() if attention_mask is not None else None
    norms = hidden.norm(dim=-1)
    return norms[mask] if mask is not None else norms.reshape(-1)


def activation_scale(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    exclude_sink: bool = True,
) -> float:
    """The scale used everywhere: median ``||h||`` over real, non-sink tokens.

    This is ``E||h||`` in PLAN.md's relative-strength convention (``r = ||alpha*v|| / E||h||``)
    and the divisor `encode` uses -- one number, shared, so a steering push of a given `r` and
    a denoiser trained at a given corruption level agree on what "typical" means (DECISIONS D2).
    """
    norms = token_norms(hidden, attention_mask, exclude_sink)
    if norms.numel() == 0:
        raise ValueError(
            "no tokens left after masking/sink-exclusion; cannot compute a scale. Check the "
            "attention mask, or that the sequence has more than just the sink position."
        )
    return float(norms.median())


def encode(hidden: torch.Tensor, scale: float, center: bool = True) -> torch.Tensor:
    """Map a raw activation into the space the denoiser trains and infers in.

    Order matters only in that it must be the same order everywhere: center first (a
    scale-independent projection), then divide by ``scale``. Forgetting the corresponding
    `decode` multiply is the un-scale trap named in CLAUDE.md -- it produces a plausible-looking
    but wrong Pareto curve, not a crash, which is why the scale is always an explicit argument
    here rather than an assumed global.
    """
    if center:
        hidden = _center(hidden)
    return hidden / scale


def decode(encoded: torch.Tensor, scale: float, center: bool = True) -> torch.Tensor:
    """Inverse of `encode`, up to centering (see module docstring: not the identity on `h`)."""
    return encoded * scale

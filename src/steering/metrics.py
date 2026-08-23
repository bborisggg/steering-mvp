"""The cheap quality axis (reference NLL, distinct-n, repetition rate) and the cheap concept
axis (``sae_concept_score``).

**Never report NLL/perplexity alone.** Degenerate repetition scores a *low* perplexity -- a
model stuck in a loop is easy to predict -- so a method that only improves NLL may simply be
collapsing into repetition rather than becoming more fluent. distinct-n and repetition rate
catch what NLL cannot see, and PLAN.md requires all of them side by side, always. This is not a
hypothetical: Step 1's own gate hit it directly (`DEVLOG.md`, 2026-08-22) -- `ppl_ratio` alone
looked heavy-tailed and inconsistent across features at matched `r`, which was resolved only by
looking at `dist_2` alongside it.

**NLL and concept score must both be scored under the unsteered model.** Neither function
defends against a hook left active on ``model`` -- see the test suite for a demonstration of
why that matters: the same text scores differently with and without one. Every call site is
responsible for passing the clean model.

**``sae_concept_score`` is not the headline concept metric.** It shares an SAE with whatever
produced the steering direction, so it is circular by construction -- useful for Step 2's
usability screen (DECISIONS D1) and for mechanism analysis, not for a final concept-vs-fluency
claim, which needs an independent judge (PLAN.md §4).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn

from steering import spaces
from steering.hooks import ResidualHook


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: list[str], n: int) -> float:
    """Unique n-grams / total n-grams, pooled across all texts.

    Low values mean repetitive generations, within or across samples -- the failure mode NLL
    alone cannot see (a loop is *easy* to predict, so it scores well on NLL).
    """
    total, unique = 0, set()
    for text in texts:
        grams = _ngrams(text.split(), n)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def repetition_rate(texts: list[str], n: int = 4) -> float:
    """Mean, per text, of the fraction of n-grams that repeat within that same text.

    Complements distinct_n, which pools across samples and so can look healthy even when every
    individual generation is its own loop of a different phrase.
    """
    rates = []
    for text in texts:
        grams = _ngrams(text.split(), n)
        if not grams:
            continue
        counts = Counter(grams)
        repeated = sum(c for g, c in counts.items() if c > 1)
        rates.append(repeated / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


@torch.no_grad()
def reference_nll(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    continuations: list[str],
    device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-example mean NLL of ``continuations[i]`` given ``prompts[i]``, under ``model`` as-is.

    Per-example, not pooled into one scalar: PLAN.md's paired bootstrap and matched-fluency
    analysis need individual values to resample, which a single corpus-level number cannot
    provide after the fact.

    Prompt tokens are masked out of the loss; only continuation tokens count. Pads right,
    managed locally regardless of the caller's current setting (see ``generate.py``'s docstring
    for why nothing here may assume a global convention). Returns NaN for any example whose
    continuation tokenizes to nothing -- returning 0.0 would read as a perfect prediction, the
    opposite of what an empty continuation means.
    """
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must be the same length")

    results: list[torch.Tensor] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            p_batch = prompts[start : start + batch_size]
            c_batch = continuations[start : start + batch_size]
            full_texts = [p + c for p, c in zip(p_batch, c_batch, strict=True)]

            prompt_lens = [
                len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in p_batch
            ]
            enc = tokenizer(
                full_texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            input_ids = enc["input_ids"]
            attention = enc["attention_mask"].bool()
            batch_len = input_ids.shape[1]

            positions = torch.arange(batch_len, device=input_ids.device).unsqueeze(0)
            plen = torch.tensor(prompt_lens, device=input_ids.device).unsqueeze(1)
            is_continuation = (positions >= plen) & attention  # [B, L], right-padded

            logits = model(**enc).logits
            shift_logits = logits[:, :-1, :].float()
            shift_targets = input_ids[:, 1:]
            shift_mask = is_continuation[:, 1:]

            logprobs = torch.log_softmax(shift_logits, dim=-1)
            token_nll = -logprobs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
            token_nll = token_nll * shift_mask

            counts = shift_mask.sum(dim=1)
            summed = token_nll.sum(dim=1)
            per_example = torch.where(
                counts > 0, summed / counts.clamp_min(1), torch.full_like(summed, float("nan"))
            )
            results.append(per_example.cpu())
    finally:
        tokenizer.padding_side = original_padding_side

    return torch.cat(results)


@torch.no_grad()
def sae_concept_score(
    model: nn.Module,
    tokenizer,
    sae,
    feature_id: int,
    layer: int,
    prompts: list[str],
    continuations: list[str],
    device,
    center: bool,
    batch_size: int = 8,
) -> dict[str, float]:
    """Does ``feature_id`` fire on the generated text? A judge-free, cheap concept axis.

    Measured under **no intervention** -- asks whether the concept reached the text, not
    whether the feature can be forced on directly. See the module docstring for why this is
    not the headline concept metric.

    ``center`` should come from ``spaces.should_center(model_name)``, matching every other
    function that reads activations in this project. Continuation tokens only, sink excluded
    (mirrors ``reference_nll``'s masking exactly, for the same reason).
    """
    from steering.vectors import sae_encode

    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must be the same length")

    fire_count, act_sum, token_count = 0, 0.0, 0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            p_batch = prompts[start : start + batch_size]
            c_batch = continuations[start : start + batch_size]
            full_texts = [p + c for p, c in zip(p_batch, c_batch, strict=True)]

            prompt_lens = [
                len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in p_batch
            ]
            enc = tokenizer(
                full_texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            input_ids = enc["input_ids"]
            attention = enc["attention_mask"].bool()
            batch_len = input_ids.shape[1]

            positions = torch.arange(batch_len, device=input_ids.device).unsqueeze(0)
            plen = torch.tensor(prompt_lens, device=input_ids.device).unsqueeze(1)
            keep = (positions >= plen) & attention
            keep[:, 0] = False  # never the sink, even in the degenerate empty-prompt case

            if not keep.any():
                continue

            with ResidualHook(model, layer=layer, capture=True) as hook:
                model(**enc)
            hidden = hook.captured[0]  # ResidualHook captures to CPU regardless of `device`
            if center:
                hidden = spaces.center(hidden)

            # `keep` was built from `enc`/`input_ids`, which live on `device`; move the mask
            # rather than the (larger) hidden states. Then move the small masked result to
            # the SAE's own device (`sae.W_dec.device`), not necessarily `device` either.
            acts = sae_encode(sae, hidden[keep.to(hidden.device)].to(sae.W_dec.device))
            acts = acts[:, feature_id]
            fire_count += int((acts > 0).sum().item())
            act_sum += float(acts.sum().item())
            token_count += acts.numel()
    finally:
        tokenizer.padding_side = original_padding_side

    if token_count == 0:
        return {"fire_rate": float("nan"), "mean_act": float("nan")}
    return {"fire_rate": fire_count / token_count, "mean_act": act_sum / token_count}


# ==================================================================================================
# Step 9B: what does the denoiser repair?
# ==================================================================================================
# Two independent diagnostics, deliberately not combined into one number: how far an activation
# sits from the clean distribution (PCA-distance), and how much of what the denoiser removes
# points along the steering direction versus everywhere else (the M3 mechanism check). Neither
# assumes a low-dimensional manifold -- this project's own lineage measured the residual stream
# needing 66-75% of its dimensions for 95% variance, so "steering pushes activations off a
# manifold" is a metaphor, not the geometry actually observed. PCAFit is full-rank accordingly.


@dataclass
class PCAFit:
    """A clean activation distribution's mean and eigendecomposed covariance, computed once so
    :meth:`distance` is a projection and a sum, not a repeated d x d eigendecomposition.

    ``ridge`` floors the smallest eigenvalues (relative to the mean eigenvalue, so it scales
    with whatever units the activations are in) before they are used as divisors -- an
    empirical covariance from a finite sample is never exactly singular in theory but can be
    close enough in practice to blow up the distance along a near-unvisited direction.
    """

    mean: torch.Tensor
    eigvecs: torch.Tensor  # [d, d], columns are eigenvectors, descending eigenvalue order
    eigvals: torch.Tensor  # [d], descending, ridge-floored

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance of each row of ``x`` to the fitted clean distribution."""
        coords = (x - self.mean) @ self.eigvecs
        return (coords.pow(2) / self.eigvals).sum(dim=-1).sqrt()

    def to(self, device) -> PCAFit:
        """Move the fit to ``device``. ``fit_pca_distance`` is typically called on the cached
        activation pool, which is always CPU (``cache_activations``'s own convention) -- but
        the points later scored against it (steered, denoised) usually live on the model's
        device. Moves only the fitted ``(mean, eigvecs, eigvals)``, not the pool that produced
        them, which is the cheap direction (``d`` and ``d x d``, not ``n x d``)."""
        return PCAFit(
            mean=self.mean.to(device), eigvecs=self.eigvecs.to(device),
            eigvals=self.eigvals.to(device),
        )


def fit_pca_distance(activations: torch.Tensor, ridge: float = 1e-2) -> PCAFit:
    """Fit :class:`PCAFit` on a pool of clean activations (Step 9B)."""
    mean = activations.mean(dim=0)
    centered = activations - mean
    cov = (centered.T @ centered) / (activations.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
    eigvals = eigvals.clamp_min(ridge * eigvals.mean())
    return PCAFit(mean=mean, eigvecs=eigvecs, eigvals=eigvals)


@torch.no_grad()
def denoiser_correction_along_v(
    denoiser_model, steered: torch.Tensor, v: torch.Tensor, t: torch.Tensor | float
) -> dict[str, torch.Tensor]:
    """Decompose what a denoiser removes from an already-steered activation into the component
    along the steering direction versus everything orthogonal to it (Step 9B).

    ``correction = steered - D(steered, t)`` -- what denoising takes away. A denoiser that is
    genuinely *erasing the steering* removes a large, positive component along ``v`` (pulling
    the activation back toward where it was before the push); one with nothing to say about the
    steering direction removes almost none of it (``fraction_along_v`` near zero) -- the exact
    mechanism the exploratory repo's own M3 result rested on: there, the correction was 97.7%
    orthogonal to `v`, which is why restoring the parallel component changed almost nothing.
    Do not assume the sign transfers here: PLAN.md itself notes GPT-2 and Gemma disagreed.

    Returns per-row tensors, not a reduced mean -- matching this module's convention elsewhere,
    since reducing here would throw away exactly the spread a paired analysis needs.
    """
    v_hat = v / v.norm()
    denoised = denoiser_model(steered, t=t)
    correction = steered - denoised
    along_v = correction @ v_hat  # [N], signed
    orthogonal = correction - along_v.unsqueeze(-1) * v_hat
    correction_norm = correction.norm(dim=-1)
    return {
        "along_v": along_v,
        "orthogonal_norm": orthogonal.norm(dim=-1),
        "correction_norm": correction_norm,
        "fraction_along_v": along_v.pow(2) / correction_norm.pow(2).clamp_min(1e-12),
    }

"""The cheap quality axis: reference NLL, distinct-n, repetition rate.

**Never report NLL/perplexity alone.** Degenerate repetition scores a *low* perplexity -- a
model stuck in a loop is easy to predict -- so a method that only improves NLL may simply be
collapsing into repetition rather than becoming more fluent. distinct-n and repetition rate
catch what NLL cannot see, and PLAN.md requires all of them side by side, always.

**NLL must be scored under the unsteered model.** ``reference_nll`` takes whatever ``model`` it
is given and does not defend against a hook left active on it -- see the test suite for a
demonstration of why that matters: the same text scores differently with and without one. Every
call site is responsible for passing the clean model.
"""

from __future__ import annotations

from collections import Counter

import torch
from torch import nn


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

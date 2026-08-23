"""Paired, reproducible generation with an intervention installed at the frozen layer.

"Paired" means every arm (B0, B1, a denoised arm later) is generated from the same prompts with
the same seed and the same decoding, so a difference in output is attributable to the
intervention and not to a resampled decode path. "Reproducible" means the seed is reset per
batch so the same call always returns the same text -- it does **not** mean greedy.

Default decoding is ancestral sampling at ``temperature=1.0, top_p=1.0, top_k=0`` (no
truncation, ``do_sample=True``), not greedy. Measured while reproducing PLAN.md Step 1's gate:
greedy decoding on GPT-2 is repetition-prone even on a completely *unsteered* continuation --
dist_2 0.66 and repetition_4 0.19 on the reference 32-prompt set, against 0.94/0.005 under
seeded sampling. A degenerate baseline leaves almost no dynamic range to show steering's actual
cost, which is the entire point of the Pareto front -- so greedy is available
(``temperature=0.0``) but is not the default. Fixed seeding makes sampled generation exactly as
reproducible from a committed config as greedy would be; it just isn't the same kind of
"deterministic" PLAN.md's requirement first suggested.

**``top_k`` defaults to 0 (disabled) deliberately, not left unset.** HF's own generation-config
resolution silently fills an *unset* ``top_k`` in to ``50`` whenever ``do_sample=True`` --
confirmed via ``model._prepare_generation_config`` -- even though ``GenerationConfig()``'s own
bare default reports ``top_k=None``. That is a fourth, undocumented restriction on top of
temperature and top_p, and it materially suppresses self-perplexity: on the reference 32-prompt
set, B0's own measured ppl was 15.9 with `top_k` left unset vs. 57+ with it explicitly disabled
-- and a suppressed baseline inflates every ppl_ratio computed against it, since the same
absolute cost from steering reads as a much larger relative multiple of a smaller denominator.
Passing ``top_k`` explicitly is what makes ``temperature=1.0, top_p=1.0`` actually mean
"unrestricted", which is what those two parameters alone imply to a reader.

Left-padding, managed locally rather than trusted from the caller. Left-padding is what makes
``generated[:, prompt_len:]`` slice out exactly the continuation for every row in a batch of
different-length prompts -- under right-padding the same slice would include trailing prompt
padding for the shorter rows. This is the mirror image of ``vectors.compute_feature_stats``,
which needs right-padding for its own reason (position 0 = sink); neither function may leave
``padding_side`` changed for the other, since nothing in this project sets it once globally.

**Every batch is grouped to a single exact tokenized length -- no batch is ever padded.** Found
the hard way, in a real Step 8 run: under left-padding, position 0 is a genuine token only for
the batch's *longest* prompt; every shorter row has a pad token there instead, with its real
first token (the attention sink -- every ``Intervention``'s "skip the sink" policy assumes
index 0 is it) sitting further in. Every intervention's own skip-the-sink logic then misses the
real sink for those rows entirely, letting a steering push -- or worse, a denoiser that has
never seen anything near the sink's ~30-40x-median magnitude -- reach it directly. The result
was catastrophic, direction-independent word salad, not a subtle shift: mean test-set
perplexity above 15,000 against a few hundred for the non-denoised arms, at every steering
strength. Grouping by exact length (not merely similar length -- "sequence bucketing" is the
standard technique, but its usual form still tolerates some padding within a bucket, which is
exactly what this project's positional correctness requirement cannot) means
``tokenizer(..., padding=True)`` is a no-op on every batch it is asked to build: there is
nothing to pad, so position 0 is the true sink for every row, always.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from steering.hooks import ResidualHook
from steering.interventions import Intervention


def _group_by_exact_length(tokenizer, prompts: list[str]) -> dict[int, list[int]]:
    """Original indices of ``prompts``, grouped by exact tokenized length.

    Insertion order within each group is preserved, so chunking a group into batches later
    stays deterministic and reproducible run to run.
    """
    groups: dict[int, list[int]] = {}
    for i, prompt in enumerate(prompts):
        length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        groups.setdefault(length, []).append(i)
    return groups


@dataclass
class GenerationResult:
    texts: list[str]
    n_decode_steps: list[int]  # per prompt-batch, not per prompt -- see generate()'s docstring


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    intervention: Intervention,
    layer: int,
    device,
    max_new_tokens: int = 48,
    batch_size: int = 16,
    seed: int = 0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> GenerationResult:
    """Generate continuations with ``intervention`` active at ``layer``.

    ``temperature=0.0`` selects greedy decoding (``do_sample=False``); any positive temperature
    samples. ``top_k=0`` means unrestricted (HF's convention); see the module docstring for why
    this is passed explicitly rather than left for HF to fill in.

    Returns only the continuation, prompt stripped, so metrics score the generated part alone.
    The seed is reset per batch so a condition's output depends on the intervention and the
    prompts, not on how they happened to be chunked into batches, nor on draws consumed by an
    earlier batch -- and any per-token state the intervention itself carries (a decaying push,
    say) is reset the same way.

    ``n_decode_steps`` has one entry per *batch*, not per prompt: the hook's call trace is
    shared across every prompt in a batch (one forward pass covers all of them), so per-prompt
    counts do not exist. It exists to make the reach-check below inspectable, not as a metric.
    """
    texts: list[str | None] = [None] * len(prompts)
    decode_steps: list[int] = []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        length_groups = _group_by_exact_length(tokenizer, prompts)
        batch_counter = 0
        for length in sorted(length_groups):
            indices = length_groups[length]
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = [prompts[i] for i in batch_indices]
                # padding=True is a no-op here: every prompt in this batch tokenizes to
                # exactly `length`, by construction of the grouping above.
                enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
                torch.manual_seed(seed + batch_counter)
                batch_counter += 1
                if hasattr(intervention, "reset"):
                    intervention.reset()

                with ResidualHook(model, layer=layer, fn=intervention) as hook:
                    generated = model.generate(
                        **enc, max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p if temperature > 0 else None,
                        top_k=top_k if temperature > 0 else None,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                if hook.n_decode_steps == 0 and max_new_tokens > 1:
                    raise RuntimeError(
                        f"intervention fired on {hook.n_calls} forward pass(es) but reached no "
                        f"decode steps for {max_new_tokens} requested tokens; it did not reach "
                        f"generated positions (see hooks.py's seq_lens)"
                    )
                decode_steps.append(hook.n_decode_steps)

                prompt_len = enc["input_ids"].shape[1]
                continuation_ids = generated[:, prompt_len:]
                decoded = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
                for i, text in zip(batch_indices, decoded, strict=True):
                    texts[i] = text
    finally:
        tokenizer.padding_side = original_padding_side

    assert all(t is not None for t in texts), "every prompt should have been covered exactly once"
    return GenerationResult(texts=texts, n_decode_steps=decode_steps)

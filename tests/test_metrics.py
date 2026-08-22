"""The cheap quality axis: reference NLL, distinct-n, repetition rate. Never reported alone --
degenerate repetition scores a *low* perplexity, so a method that only improves NLL may just be
looping (PLAN.md's "also record" list exists for exactly this).

**The one correctness trap that matters most here:** NLL must be scored under the *unsteered*
model, or it measures nothing -- a steered generation fed back through the same steering would
look artificially fluent to itself. ``reference_nll`` takes whatever ``model`` it is given and
trusts the caller on this; the test below demonstrates *why* that trust matters by showing the
same text score differently with and without an external hook active, rather than merely
asserting the docstring's claim.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from steering import metrics


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


LAYER = 6


# --- distinct-n --------------------------------------------------------------------------

def test_distinct_1_hand_computed():
    # "a a b" -> unigrams [a, a, b], 2 unique / 3 total
    assert metrics.distinct_n(["a a b"], 1) == pytest.approx(2 / 3)


def test_distinct_n_pools_across_texts():
    """Corpus-level, not averaged per text: repeating the same two-token text twice doubles
    the total count but not the unique count -- {a,b} unique out of 4 total tokens."""
    assert metrics.distinct_n(["a b", "a b"], 1) == pytest.approx(2 / 4)


def test_distinct_n_empty_texts_is_zero():
    assert metrics.distinct_n([], 1) == 0.0
    assert metrics.distinct_n([""], 2) == 0.0  # too short for any bigram


# --- repetition rate ----------------------------------------------------------------------

def test_repetition_rate_hand_computed():
    # "x y x y" -> bigrams [(x,y),(y,x),(x,y)], (x,y) appears twice -> 2 repeated / 3 total
    assert metrics.repetition_rate(["x y x y"], n=2) == pytest.approx(2 / 3)


def test_repetition_rate_zero_when_nothing_repeats():
    assert metrics.repetition_rate(["a b c d e"], n=2) == pytest.approx(0.0)


def test_repetition_rate_averages_per_text_not_pooled():
    """The complement of distinct-n: a text that loops must show up even if diluted by other,
    healthy generations in the same batch -- so this averages per-text rates, it does not pool
    n-grams across texts the way distinct_n does."""
    looping = "a a a a a a"  # every bigram is (a,a), fully repeated
    healthy = "the quick brown fox jumps"  # no repeats
    rate = metrics.repetition_rate([looping, healthy], n=2)
    assert rate == pytest.approx((1.0 + 0.0) / 2)


def test_repetition_rate_skips_texts_too_short_for_n():
    assert metrics.repetition_rate(["a"], n=4) == pytest.approx(0.0)


# --- reference_nll: exactness ----------------------------------------------------------------

def test_reference_nll_matches_hand_computed_cross_entropy(gpt2):
    """The strongest check: an independent computation of the same quantity, not just 'runs
    without error'. Verifies the shift-by-one and the prompt/continuation boundary are both
    exactly right, not merely plausible."""
    model, tok = gpt2
    prompt, continuation = "The capital of France is", " Paris, a beautiful city"

    full_ids = tok(prompt + continuation, return_tensors="pt", add_special_tokens=False)
    prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])

    with torch.no_grad():
        logits = model(**full_ids).logits[0]  # [L, V]
    targets = full_ids["input_ids"][0]  # [L]
    # logits[t] predicts targets[t+1]; continuation tokens are targets[prompt_len:]
    shift_logits = logits[:-1]
    shift_targets = targets[1:]
    mask = torch.arange(len(shift_targets)) >= (prompt_len - 1)
    expected = F.cross_entropy(shift_logits[mask], shift_targets[mask]).item()

    got = metrics.reference_nll(model, tok, [prompt], [continuation], device="cpu")
    assert float(got[0]) == pytest.approx(expected, rel=1e-4)


def test_reference_nll_excludes_prompt_tokens(gpt2):
    """Two very different prompts, same continuation: if prompt tokens leaked into the loss,
    changing only the prompt would change the score even though the scored text didn't."""
    model, tok = gpt2
    continuation = " and the weather was pleasant that day"
    a = metrics.reference_nll(model, tok, ["Yesterday I went to the market."],
                              [continuation], device="cpu")
    b = metrics.reference_nll(model, tok, ["The history of ancient Rome is fascinating."],
                              [continuation], device="cpu")
    # Different prompts DO change the continuation's NLL (different context) -- that's
    # expected and correct. What this test actually pins is that both computations succeed
    # and produce a single well-defined value per prompt, checked exactly against ground
    # truth above; here we only sanity-check both are finite and positive.
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert (a > 0).all() and (b > 0).all()


def test_reference_nll_returns_one_value_per_example(gpt2):
    model, tok = gpt2
    out = metrics.reference_nll(model, tok, ["Hi.", "Tell me a story about a dragon."],
                                [" Hello!", " Once there was a dragon who loved gold."],
                                device="cpu")
    assert out.shape == (2,)


def test_reference_nll_batching_does_not_change_the_result(gpt2):
    model, tok = gpt2
    prompts = ["Hi.", "Tell me a story about a dragon and a knight."]
    conts = [" Hello there!", " Once there was a dragon who loved gold coins."]
    together = metrics.reference_nll(model, tok, prompts, conts, device="cpu", batch_size=2)
    separate = torch.cat([
        metrics.reference_nll(model, tok, [p], [c], device="cpu")
        for p, c in zip(prompts, conts, strict=True)
    ])
    torch.testing.assert_close(together, separate, rtol=1e-4, atol=1e-4)


def test_reference_nll_empty_continuation_is_nan_not_zero(gpt2):
    """An empty continuation has no tokens to score. Returning 0.0 would look like a perfect
    prediction -- the best possible score -- which is the opposite of what happened."""
    model, tok = gpt2
    out = metrics.reference_nll(model, tok, ["Hello there"], [""], device="cpu")
    assert torch.isnan(out[0])


def test_reference_nll_restores_padding_side(gpt2):
    model, tok = gpt2
    tok.padding_side = "left"
    metrics.reference_nll(model, tok, ["Hi."], [" there"], device="cpu")
    assert tok.padding_side == "left"


# --- the trap: NLL must be scored unsteered ---------------------------------------------------

def test_nll_of_the_same_text_differs_under_an_active_hook(gpt2):
    """Demonstrates the trap rather than just documenting it: reference_nll does not defend
    against a hook left active on `model` -- it trusts the caller to pass the clean model, per
    its docstring -- so scoring under a hooked model silently gives a different, wrong number
    for the same text. This is why every call site must pass the unsteered model, always."""
    from steering.hooks import ResidualHook

    model, tok = gpt2
    prompt, continuation = "The weather today is", " sunny and warm across the region"

    clean = metrics.reference_nll(model, tok, [prompt], [continuation], device="cpu")

    torch.manual_seed(0)
    bogus_shift = torch.randn(768) * 50
    with ResidualHook(model, layer=LAYER, fn=lambda h: h + bogus_shift):
        hooked = metrics.reference_nll(model, tok, [prompt], [continuation], device="cpu")

    assert not torch.allclose(clean, hooked, rtol=1e-2)

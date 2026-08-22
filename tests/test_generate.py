"""Paired deterministic generation with an intervention installed at the frozen layer.

Two things this exists to guarantee, both of which fail silently rather than crash if wrong:

1. **The intervention reaches generated positions**, not just the prompt -- checked against
   ``hooks.py``'s ``n_decode_steps``, which counts single-token forward passes rather than
   trusting a bare call count (a prefill-only hook can still fire once and look fine).
2. **Padding correctness under a real batch of different-length prompts.** Generation needs
   left-padding (so `generated[:, prompt_len:]` slices the continuation correctly for every row
   in a batch); this is the opposite of what `vectors.compute_feature_stats` needs, and nothing
   in this project sets a global `padding_side` once for every caller. So this function manages
   it locally, the same discipline just added to `compute_feature_stats`.
"""

from __future__ import annotations

import pytest
import torch

from steering import generate, interventions


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


LAYER = 6


# --- basic generation ------------------------------------------------------------------------

def test_returns_one_continuation_per_prompt(gpt2):
    model, tok = gpt2
    out = generate.generate(model, tok, ["The capital of France is", "In 1969, humans"],
                            interventions.NoSteering(), layer=LAYER, device="cpu",
                            max_new_tokens=8)
    assert len(out.texts) == 2
    assert all(isinstance(t, str) and t for t in out.texts)


def test_continuation_excludes_the_prompt(gpt2):
    model, tok = gpt2
    out = generate.generate(model, tok, ["The capital of France is"], interventions.NoSteering(),
                            layer=LAYER, device="cpu", max_new_tokens=8)
    assert "capital of France" not in out.texts[0]


def test_is_deterministic_across_repeated_calls(gpt2):
    """Reproducible from a config means a fixed seed reproduces exactly -- under sampling too,
    not only under greedy. This is the module's actual "deterministic" guarantee."""
    model, tok = gpt2
    a = generate.generate(model, tok, ["Tell me about your day."], interventions.NoSteering(),
                          layer=LAYER, device="cpu", max_new_tokens=12, seed=0)
    b = generate.generate(model, tok, ["Tell me about your day."], interventions.NoSteering(),
                          layer=LAYER, device="cpu", max_new_tokens=12, seed=0)
    assert a.texts == b.texts


def test_different_seeds_sample_different_text(gpt2):
    model, tok = gpt2
    a = generate.generate(model, tok, ["Tell me about your day."], interventions.NoSteering(),
                          layer=LAYER, device="cpu", max_new_tokens=12, seed=0)
    b = generate.generate(model, tok, ["Tell me about your day."], interventions.NoSteering(),
                          layer=LAYER, device="cpu", max_new_tokens=12, seed=1)
    assert a.texts != b.texts


def test_temperature_zero_is_greedy_and_matches_argmax_decoding(gpt2):
    model, tok = gpt2
    a = generate.generate(model, tok, ["The capital of France is"], interventions.NoSteering(),
                          layer=LAYER, device="cpu", max_new_tokens=8, temperature=0.0, seed=0)
    b = generate.generate(model, tok, ["The capital of France is"], interventions.NoSteering(),
                          layer=LAYER, device="cpu", max_new_tokens=8, temperature=0.0, seed=99)
    assert a.texts == b.texts, "greedy must ignore the seed entirely -- no randomness to seed"


def test_top_k_is_passed_explicitly_not_left_for_hf_to_default(gpt2, monkeypatch):
    """The bug this session found: HF's generation-config resolution silently fills an unset
    ``top_k`` in to 50 whenever ``do_sample=True`` (confirmed via
    ``model._prepare_generation_config``), even though ``GenerationConfig()``'s own bare
    default reports ``top_k=None``. That is a real, undocumented restriction contradicting what
    "temperature=1.0, top_p=1.0" is supposed to mean. Checked by spying on the actual call."""
    model, tok = gpt2
    seen = {}
    original_generate = model.generate

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(model, "generate", spy)
    generate.generate(model, tok, ["Hello"], interventions.NoSteering(), layer=LAYER,
                      device="cpu", max_new_tokens=3)
    assert seen["top_k"] == 0


def test_disabling_top_k_measurably_raises_self_perplexity(gpt2):
    """The consequence, not just the config value: with top_k left at HF's silent default of
    50, GPT-2's own unsteered samples score as far more predictable to itself than they
    actually are under unrestricted sampling -- a suppressed baseline that then inflates every
    ppl_ratio computed against it. Small sample (8 prompts) for test speed; the effect is large
    enough to show clearly even so (measured on the full 32-prompt reference set: 15.9 vs 57+)."""
    from steering import metrics

    model, tok = gpt2
    prompts = ["The best thing about", "Yesterday I went to", "She looked at the",
               "In the morning, the", "It is well known that", "The city was",
               "After a long day,", "Scientists have discovered"]

    restricted = generate.generate(model, tok, prompts, interventions.NoSteering(), layer=LAYER,
                                   device="cpu", max_new_tokens=24, seed=0, top_k=50)
    unrestricted = generate.generate(model, tok, prompts, interventions.NoSteering(), layer=LAYER,
                                     device="cpu", max_new_tokens=24, seed=0, top_k=0)

    nll_r = metrics.reference_nll(model, tok, prompts, restricted.texts, device="cpu")
    nll_u = metrics.reference_nll(model, tok, prompts, unrestricted.texts, device="cpu")
    ppl_r = float(nll_r[~nll_r.isnan()].mean().exp())
    ppl_u = float(nll_u[~nll_u.isnan()].mean().exp())
    assert ppl_u > ppl_r


def test_default_temperature_reproduces_healthy_baseline_fluency(gpt2):
    """Regression test for the actual bug this session found: greedy decoding produced a
    visibly degenerate *unsteered* baseline (dist_2 0.66, repetition_4 0.19 on the reference
    32-prompt set) with almost no dynamic range left to show steering's cost. Sampling at the
    default temperature=1.0 must not reproduce that regression."""
    from steering import metrics

    model, tok = gpt2
    prompts = ["The best thing about", "Yesterday I went to", "She looked at the",
               "In the morning, the", "It is well known that"] * 4
    out = generate.generate(model, tok, prompts, interventions.NoSteering(), layer=LAYER,
                            device="cpu", max_new_tokens=24, batch_size=8, seed=0)
    assert metrics.distinct_n(out.texts, 2) > 0.8
    assert metrics.repetition_rate(out.texts, 4) < 0.05


# --- the intervention must reach generated positions ------------------------------------------

def test_steering_changes_the_continuation(gpt2):
    model, tok = gpt2
    torch.manual_seed(0)
    v = torch.randn(768)
    b1 = interventions.AdditiveSteering(v, r=2.0, scale=90.0)
    clean = generate.generate(model, tok, ["The weather today is"], interventions.NoSteering(),
                              layer=LAYER, device="cpu", max_new_tokens=15)
    steered = generate.generate(model, tok, ["The weather today is"], b1,
                                layer=LAYER, device="cpu", max_new_tokens=15)
    assert clean.texts != steered.texts


def test_records_decode_steps_reached(gpt2):
    """This is the actual guarantee, not just 'text changed' -- see module docstring."""
    model, tok = gpt2
    out = generate.generate(model, tok, ["Once upon a time"], interventions.NoSteering(),
                            layer=LAYER, device="cpu", max_new_tokens=10)
    assert out.n_decode_steps[0] >= 1


def test_raises_if_the_intervention_never_reaches_decode(gpt2, monkeypatch):
    """The guard's own branch, isolated: a hook that genuinely reports zero decode steps must
    raise, tested directly rather than by trying to make a real hook misbehave (real forward
    hooks fire unconditionally whenever their block runs, so there is no legitimate way for a
    broken *intervention* callable to prevent that -- this guard is defence in depth, and its
    logic is what is under test here, not hooks.py's mechanics, which are covered separately)."""
    model, tok = gpt2

    class FakeHook:
        def __init__(self, model, layer, fn):
            self.n_calls = 1
            self.n_decode_steps = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(generate, "ResidualHook", FakeHook)
    with pytest.raises(RuntimeError, match="did not reach"):
        generate.generate(model, tok, ["Hello there"], interventions.NoSteering(),
                          layer=LAYER, device="cpu", max_new_tokens=5)


# --- batching and padding -------------------------------------------------------------------

def test_batching_does_not_change_the_result_under_greedy(gpt2):
    """Padding correctness, isolated from decoding strategy: a short and a long prompt batched
    together must generate exactly what they would alone, or left-padding is misaligning the
    continuation slice. Pinned to greedy (temperature=0) deliberately -- see
    test_sampling_is_not_batch_invariant below for why this would be the wrong invariant to
    check under sampling."""
    model, tok = gpt2
    short, long_ = "Hi.", "Tell me a detailed story about a brave knight"
    together = generate.generate(model, tok, [short, long_], interventions.NoSteering(),
                                 layer=LAYER, device="cpu", max_new_tokens=10, batch_size=2,
                                 temperature=0.0)
    alone_short = generate.generate(model, tok, [short], interventions.NoSteering(),
                                    layer=LAYER, device="cpu", max_new_tokens=10, temperature=0.0)
    alone_long = generate.generate(model, tok, [long_], interventions.NoSteering(),
                                   layer=LAYER, device="cpu", max_new_tokens=10, temperature=0.0)
    assert together.texts[0] == alone_short.texts[0]
    assert together.texts[1] == alone_long.texts[0]


def test_sampling_is_not_batch_invariant_even_with_a_fixed_seed(gpt2):
    """The caveat the test above exists to isolate away from: under sampling, a batch draws its
    random numbers jointly across rows at each step, in an order that depends on what else is
    in the batch. So resetting the same seed per batch makes a *given* batch_size fully
    reproducible run to run, but it does **not** make one prompt's output independent of which
    other prompts happen to share its batch -- unlike greedy, where no randness is drawn and
    batch composition genuinely cannot matter. Documented here so it is not mistaken for a bug
    later (e.g. Step 8's per-example bootstrap re-running at a different batch_size)."""
    model, tok = gpt2
    short, long_ = "Hi.", "Tell me a detailed story about a brave knight"
    together = generate.generate(model, tok, [short, long_], interventions.NoSteering(),
                                 layer=LAYER, device="cpu", max_new_tokens=10, batch_size=2)
    alone_short = generate.generate(model, tok, [short], interventions.NoSteering(),
                                    layer=LAYER, device="cpu", max_new_tokens=10)
    assert together.texts[0] != alone_short.texts[0]


def test_restores_padding_side(gpt2):
    model, tok = gpt2
    tok.padding_side = "right"
    generate.generate(model, tok, ["Hello"], interventions.NoSteering(), layer=LAYER,
                      device="cpu", max_new_tokens=5)
    assert tok.padding_side == "right"


def test_uses_left_padding_internally_regardless_of_caller_state(gpt2):
    """The opposite fixture to compute_feature_stats's: whatever the caller left padding_side
    set to, generation must still align continuations correctly across a mixed-length batch.
    Greedy, for the same reason as the test above -- this checks alignment, not sampling."""
    model, tok = gpt2
    tok.padding_side = "right"  # deliberately the wrong side for generation
    short, long_ = "Hi.", "Tell me a detailed story about a brave knight"
    out = generate.generate(model, tok, [short, long_], interventions.NoSteering(),
                            layer=LAYER, device="cpu", max_new_tokens=10, batch_size=2,
                            temperature=0.0)
    alone_short = generate.generate(model, tok, [short], interventions.NoSteering(),
                                    layer=LAYER, device="cpu", max_new_tokens=10, temperature=0.0)
    assert out.texts[0] == alone_short.texts[0]


# --- intervention reset between batches -------------------------------------------------------

def test_resets_a_stateful_intervention_once_per_batch(gpt2):
    """A position-dependent intervention (like the exploratory repo's decaying steering) must
    not carry state across batch boundaries, or a run's output depends on the batch_size it
    happened to be called with -- an invisible, unreproducible confound. Exact count, not a
    loose bound: 4 prompts at batch_size=2 is exactly 2 batches, so reset() must fire exactly
    twice -- not zero (never reset) and not once per prompt (reset in the wrong place)."""
    model, tok = gpt2

    class CountingIntervention(interventions.Intervention):
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def apply(self, hidden):
            return hidden

    iv = CountingIntervention()
    generate.generate(model, tok, ["a", "b", "c", "d"], iv, layer=LAYER, device="cpu",
                      max_new_tokens=3, batch_size=2)
    assert iv.reset_count == 2

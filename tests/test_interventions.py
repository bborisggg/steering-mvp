"""B0, B1, B2 -- the interventions Step 1's gate needs, nothing past it yet.

Deliberately named after PLAN.md's own convention, not the exploratory repo's: PLAN.md defines
``r = ||alpha*v|| / E||h||`` and calls ``r`` the thing that gets swept (the grid in PLAN.md's
alpha section is a grid of ``r``, e.g. 0, 0.1, ... 3.0). The exploratory repo confusingly named
its *own* constructor argument ``alpha`` when it meant this same relative quantity. Following
PLAN.md's vocabulary here rather than porting the name avoids reproducing that confusion.

Every intervention skips the attention sink (DECISIONS D4, decided this session): position 0 is
touched by nothing, steering or denoising alike. The mechanism is the one already proven in
hooks.py's prefill-only test -- seq_len > 1 identifies exactly the prefill pass, so "skip index 0
when seq_len > 1" skips exactly the sink, with no separate position bookkeeping to keep in sync
with the KV cache.
"""

from __future__ import annotations

import pytest
import torch

from steering import interventions

D_MODEL = 8


@pytest.fixture
def v():
    torch.manual_seed(0)
    return torch.randn(D_MODEL)  # deliberately not unit-norm, to check normalisation happens


@pytest.fixture
def prompt_hidden():
    """[batch=2, seq=5, d_model], standing in for a prefill pass (seq_len > 1)."""
    torch.manual_seed(1)
    return torch.randn(2, 5, D_MODEL) * 90  # roughly GPT-2 layer-6 scale


@pytest.fixture
def decode_hidden():
    """[batch=2, seq=1, d_model], standing in for a single decode step."""
    torch.manual_seed(2)
    return torch.randn(2, 1, D_MODEL) * 90


# --- B0 ----------------------------------------------------------------------------------

def test_b0_is_identity_on_a_prefill_pass(prompt_hidden):
    out = interventions.NoSteering()(prompt_hidden)
    result = prompt_hidden if out is None else out
    torch.testing.assert_close(result, prompt_hidden)


def test_b0_is_identity_including_the_sink_position(prompt_hidden):
    """B0 is the reference; if it touched anything, every comparison against it would be off."""
    out = interventions.NoSteering()(prompt_hidden)
    result = prompt_hidden if out is None else out
    torch.testing.assert_close(result[:, 0, :], prompt_hidden[:, 0, :])


# --- B1: r -> alpha ------------------------------------------------------------------------

def test_b1_normalises_v_internally(v):
    b1 = interventions.AdditiveSteering(v, r=1.0, scale=100.0)
    assert b1.v.norm() == pytest.approx(1.0, rel=1e-5)


def test_b1_alpha_equals_r_times_scale(v):
    b1 = interventions.AdditiveSteering(v, r=0.5, scale=88.5)
    assert b1.alpha == pytest.approx(0.5 * 88.5)


def test_b1_delta_norm_equals_alpha(v):
    """r = ||alpha*v|| / scale by definition (PLAN.md); with unit v, ||delta|| == alpha exactly."""
    b1 = interventions.AdditiveSteering(v, r=0.7, scale=88.5)
    assert float(b1.delta.norm()) == pytest.approx(b1.alpha, rel=1e-5)


def test_b1_r_zero_is_the_identity(v, decode_hidden):
    b1 = interventions.AdditiveSteering(v, r=0.0, scale=88.5)
    out = b1(decode_hidden)
    torch.testing.assert_close(out, decode_hidden)


# --- sink skipping, D4 ---------------------------------------------------------------------

def test_b1_leaves_position_zero_untouched_on_a_prefill_pass(v, prompt_hidden):
    b1 = interventions.AdditiveSteering(v, r=1.0, scale=90.0)
    out = b1(prompt_hidden)
    torch.testing.assert_close(out[:, 0, :], prompt_hidden[:, 0, :])


def test_b1_steers_every_non_sink_position_on_a_prefill_pass(v, prompt_hidden):
    b1 = interventions.AdditiveSteering(v, r=1.0, scale=90.0)
    out = b1(prompt_hidden)
    assert not torch.allclose(out[:, 1:, :], prompt_hidden[:, 1:, :])
    expected_tail = prompt_hidden[:, 1:, :] + b1.delta
    torch.testing.assert_close(out[:, 1:, :], expected_tail)


def test_b1_steers_a_decode_step_fully(v, decode_hidden):
    """A decode step (seq_len==1) is a real generated position, not the sink -- it must be
    steered, not skipped, even though it is also the only position in its forward pass."""
    b1 = interventions.AdditiveSteering(v, r=1.0, scale=90.0)
    out = b1(decode_hidden)
    torch.testing.assert_close(out, decode_hidden + b1.delta)


# --- B2: norm-matched control ----------------------------------------------------------------

def test_b2_restores_the_original_per_token_norm(v, prompt_hidden):
    b2 = interventions.NormMatchedSteering(v, r=1.5, scale=90.0)
    out = b2(prompt_hidden)
    torch.testing.assert_close(
        out[:, 1:, :].norm(dim=-1), prompt_hidden[:, 1:, :].norm(dim=-1), rtol=1e-4, atol=1e-3
    )


def test_b2_does_not_restore_the_sink_norm_because_it_never_touches_it(v, prompt_hidden):
    b2 = interventions.NormMatchedSteering(v, r=1.5, scale=90.0)
    out = b2(prompt_hidden)
    torch.testing.assert_close(out[:, 0, :], prompt_hidden[:, 0, :])


def test_b2_points_in_exactly_the_same_direction_as_b1(v, prompt_hidden):
    """B2(h) = c * B1(h) for some scalar c > 0 -- renormalising only rescales the sum h+delta,
    it cannot rotate it. So B2 and B1 agree on cosine similarity with v exactly, not
    approximately; only their norm differs. (An earlier version of this test compared raw
    projection onto v between B2 and B1 and expected B2's to be smaller -- false in general:
    rescaling the *whole* vector back to the original norm can amplify h's own component along
    v as easily as damping it, depending on geometry. Checking direction rather than projection
    magnitude is the property that is actually guaranteed.)"""
    b2 = interventions.NormMatchedSteering(v, r=1.5, scale=90.0)
    b1 = interventions.AdditiveSteering(v, r=1.5, scale=90.0)
    out_b2, out_b1 = b2(prompt_hidden), b1(prompt_hidden)
    unit_v = (v / v.norm()).expand_as(out_b2[:, 1:, :])
    cos_b2 = torch.cosine_similarity(out_b2[:, 1:, :], unit_v, dim=-1)
    cos_b1 = torch.cosine_similarity(out_b1[:, 1:, :], unit_v, dim=-1)
    torch.testing.assert_close(cos_b2, cos_b1, rtol=1e-4, atol=1e-4)


def test_b2_moves_toward_v_relative_to_clean(v, prompt_hidden):
    b2 = interventions.NormMatchedSteering(v, r=1.5, scale=90.0)
    out = b2(prompt_hidden)
    unit_v = (v / v.norm()).expand_as(out[:, 1:, :])
    cos_after = torch.cosine_similarity(out[:, 1:, :], unit_v, dim=-1)
    cos_before = torch.cosine_similarity(prompt_hidden[:, 1:, :], unit_v, dim=-1)
    assert bool((cos_after > cos_before).all())


def test_b2_r_zero_is_the_identity(v, prompt_hidden):
    b2 = interventions.NormMatchedSteering(v, r=0.0, scale=90.0)
    out = b2(prompt_hidden)
    torch.testing.assert_close(out, prompt_hidden, rtol=1e-4, atol=1e-3)


# --- installed on a real hook, end to end ---------------------------------------------------

def test_b1_installed_on_a_real_hook_reaches_decode_steps(v):
    """The point of all of this: does it actually move GPT-2's output during generation, not
    just in isolation? Reuses the same reach-check as test_hooks.py, now through the public
    Intervention interface rather than a raw lambda."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering import hooks

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    batch = tok(["The weather today is"], return_tensors="pt")

    direction = torch.randn(768)
    b1 = interventions.AdditiveSteering(direction, r=2.0, scale=90.0)
    b0 = interventions.NoSteering()

    outputs = {}
    for name, iv in (("b1", b1), ("b0", b0)):
        with hooks.ResidualHook(model, layer=6, fn=iv):
            outputs[name] = model.generate(**batch, max_new_tokens=10, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
    assert not torch.equal(outputs["b1"], outputs["b0"])

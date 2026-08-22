"""encode/decode is the one place normalisation happens; every other module must go through it.

Two facts drive the design, both measured (not assumed) on real GPT-2 and Gemma-2-2b-it
forwards; see DEVLOG for the numbers:

1. Centering (subtracting the per-token mean along d_model) is exactly output-neutral on
   GPT-2 -- LayerNorm removes the mean before every read anyway -- and a real perturbation on
   Gemma-2, which uses RMSNorm and never subtracts a mean. So centering is gated per
   architecture, not a global switch.
2. GPT-2's residual stream carries an attention sink at position 0 with ~30-40x the median
   norm of every other position. A scale computed by mean, or a per-tensor max, would be
   dominated by one token; the median over non-sink positions is what stays interpretable as
   "typical activation size" and is what DECISIONS.md D2 pre-registers.
"""

from __future__ import annotations

import pytest
import torch

from steering import spaces


@pytest.fixture(scope="module")
def gpt2_acts():
    """Real layer-6 GPT-2 activations, not synthetic -- the sink and outlier dims are real."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering import hooks

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    batch = tok(["The capital of France is", "In 1969, humans first landed on the moon"],
                return_tensors="pt", padding=True)
    with hooks.ResidualHook(model, layer=6, capture=True) as hook, torch.no_grad():
        model(**batch)
    return hook.captured[0]  # [2, T, 768], includes the sink at position 0


# --- architecture gate -----------------------------------------------------------------------

def test_gpt2_centers_by_default():
    assert spaces.should_center("gpt2") is True


def test_gemma_does_not_center():
    """RMSNorm never subtracts the mean; centering it perturbs the model. Measured: 3.2e-2
    relative logit change on Gemma-2-2b-it, ~1e5x GPT-2's 3.4e-7 (both first-party, DEVLOG)."""
    assert spaces.should_center("google/gemma-2-2b-it") is False


def test_unknown_model_name_raises_rather_than_guessing():
    """A silent default would centre an architecture nobody checked. Wrong for RMSNorm."""
    with pytest.raises(ValueError, match="unknown architecture"):
        spaces.should_center("some-new-model-nobody-added-yet")


# --- activation_scale --------------------------------------------------------------------

def test_scale_excludes_the_sink(gpt2_acts):
    with_sink = spaces.activation_scale(gpt2_acts, exclude_sink=False)
    without_sink = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    # The sink is ~30-40x the body; including it in a median over ~10 tokens per sequence
    # still drags the estimate noticeably even though median resists outliers better than mean.
    assert without_sink < with_sink
    assert without_sink < 150, "sink-free scale should be near GPT-2's typical ~90, not the sink"


def test_scale_is_the_median_not_the_mean(gpt2_acts):
    """Median specifically: DECISIONS D2 requires it because the sink is a large enough
    outlier that even excluding position 0, remaining outlier dimensions skew a mean."""
    body = gpt2_acts[:, 1:, :].reshape(-1, gpt2_acts.shape[-1])
    norms = body.norm(dim=-1)
    scale = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    assert scale == pytest.approx(float(norms.median()), rel=1e-5)
    assert scale != pytest.approx(float(norms.mean()), rel=1e-2)


def test_scale_respects_padding_mask(gpt2_acts):
    """Padded positions must not enter the statistic, or scale drifts with batch composition."""
    d = gpt2_acts.shape[-1]
    padded = torch.cat([gpt2_acts, torch.randn(2, 5, d) * 1000], dim=1)
    mask = torch.cat([torch.ones(2, gpt2_acts.shape[1]), torch.zeros(2, 5)], dim=1).bool()
    masked_scale = spaces.activation_scale(padded, exclude_sink=True, attention_mask=mask)
    unmasked_scale = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    assert masked_scale == pytest.approx(unmasked_scale, rel=1e-5)


def test_scale_raises_when_nothing_is_left():
    single_token = torch.randn(1, 1, 768)  # only the sink position, excluded -> empty
    with pytest.raises(ValueError, match="no tokens"):
        spaces.activation_scale(single_token, exclude_sink=True)


# --- encode / decode -----------------------------------------------------------------------

def test_encode_decode_round_trips_to_the_centered_form(gpt2_acts):
    """decode(encode(h)) == center(h), not == h.

    Centering is a projection, not written back and forth losslessly: GPT-2 writes the
    centered activation into the stream permanently (safe, since it's output-neutral there),
    so there is nothing to restore on decode. The round trip is to the *canonical* form the
    model actually computes with, not to whatever the caller happened to pass in.
    """
    scale = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    encoded = spaces.encode(gpt2_acts, scale=scale, center=True)
    decoded = spaces.decode(encoded, scale=scale, center=True)
    torch.testing.assert_close(decoded, spaces.center(gpt2_acts))


def test_encode_decode_round_trips_exactly_when_center_is_off(gpt2_acts):
    scale = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    encoded = spaces.encode(gpt2_acts, scale=scale, center=False)
    decoded = spaces.decode(encoded, scale=scale, center=False)
    torch.testing.assert_close(decoded, gpt2_acts)


def test_encode_applies_the_scale(gpt2_acts):
    """Forgetting the un-scale is the correctness trap CLAUDE.md names explicitly: it produces
    a plausible-looking but wrong Pareto curve, not a crash."""
    scale = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    encoded = spaces.encode(gpt2_acts, scale=scale, center=False)
    body = encoded[:, 1:, :].reshape(-1, encoded.shape[-1])
    assert body.norm(dim=-1).median() == pytest.approx(1.0, rel=0.05), (
        "encoding by the typical-token scale should put typical tokens near unit norm"
    )


def test_decode_without_encode_scale_is_wrong_by_construction(gpt2_acts):
    """A caller using the wrong scale must get a visibly different answer, not a subtly-off
    one -- otherwise a config-hash mismatch (io.py) is the only thing standing between a
    correct run and a silently wrong one."""
    scale = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    encoded = spaces.encode(gpt2_acts, scale=scale, center=False)
    wrong = spaces.decode(encoded, scale=scale * 2, center=False)
    right = spaces.decode(encoded, scale=scale, center=False)
    assert not torch.allclose(wrong, right, rtol=0.1)


def test_encode_decode_is_a_dataclass_of_config_not_a_global(gpt2_acts):
    """Two different scales must not interfere -- this will be called with GPT-2's scale and
    Gemma's scale in the same notebook session."""
    s1 = spaces.activation_scale(gpt2_acts, exclude_sink=True)
    s2 = s1 * 3.0
    e1 = spaces.encode(gpt2_acts, scale=s1, center=False)
    e2 = spaces.encode(gpt2_acts, scale=s2, center=False)
    assert not torch.allclose(e1, e2)
    torch.testing.assert_close(spaces.decode(e1, scale=s1, center=False), gpt2_acts)
    torch.testing.assert_close(spaces.decode(e2, scale=s2, center=False), gpt2_acts)


# --- centering itself ------------------------------------------------------------------------

def test_center_removes_the_all_ones_component(gpt2_acts):
    centered = spaces.center(gpt2_acts)
    assert centered.mean(dim=-1).abs().max() < 1e-4


def test_center_is_idempotent(gpt2_acts):
    once = spaces.center(gpt2_acts)
    twice = spaces.center(once)
    torch.testing.assert_close(once, twice)


def test_center_is_a_small_correction_on_gpt2(gpt2_acts):
    """The all-ones component is real but modest (measured ~1% of ||h||) -- centering is not
    supposed to be a big edit; if it were, that would be a sign the source tensor is wrong."""
    body = gpt2_acts[:, 1:, :]
    delta = (body - spaces.center(body)).norm(dim=-1)
    assert (delta / body.norm(dim=-1)).median() < 0.05

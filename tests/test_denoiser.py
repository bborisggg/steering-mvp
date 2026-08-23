"""Activation caching for denoiser training (PLAN.md Step 4).

Deliberately stores **raw** (uncentered, unscaled) activations, not pre-normalised ones. The
denoiser's own forward pass is ``D(x, s) = x + f(N(x), e(s))`` -- the encode step lives inside
``D()``, applied via ``spaces.encode``. Baking centering/scaling into the cache would mean the
transform runs once at cache time and never again, which is the opposite of "training and
inference pass through the same transform" (PLAN.md Step 4): it should run *inside* every call
to ``D()``, at both training and inference, so a change to the transform can never desync the
two the way baking it into stored data would risk.

Uses ``ResidualHook(capture=True)``, same as ``vectors.compute_feature_stats`` and
``metrics.sae_concept_score`` -- and so inherits the same device trap those had (capture is
always CPU; a mask built from ``enc`` lives on the model's device). Fixed from the start here,
with the same MPS-gated regression test added retroactively to the other two.
"""

from __future__ import annotations

import math

import pytest
import torch

from steering import denoiser


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


LAYER = 6
TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "In 1969, humans first set foot on the surface of the Moon.",
    "She opened the door slowly, unsure of what she would find inside.",
    "Markets fell sharply today amid concerns over interest rate policy.",
] * 4


def test_returns_the_right_shape(gpt2):
    model, tok = gpt2
    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=TEXTS, max_tokens=10_000,
                                      batch_size=4, max_length=32)
    assert acts.ndim == 2
    assert acts.shape[1] == 768


def test_returns_fp32_even_when_the_model_is_loaded_in_bf16():
    """Gemma loads bf16 by default (GPT-2 never exercises this path, which is exactly why the
    bug survived to a real run): activations must come back fp32 regardless, or the first
    matmul against ResidualMLPDenoiser's fp32 weights crashes MPS with a hard Metal assertion
    (SIGABRT, no Python traceback) rather than upcasting the way CPU/CUDA silently would.
    Cast a GPT-2 copy to bf16 here so this is caught on CPU, without needing Gemma in the
    test suite.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to(torch.bfloat16)

    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=TEXTS, max_tokens=200,
                                      batch_size=4, max_length=32)
    assert acts.dtype == torch.float32


def test_stops_at_max_tokens(gpt2):
    model, tok = gpt2
    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=TEXTS, max_tokens=50,
                                      batch_size=4, max_length=32)
    assert acts.shape[0] <= 50
    assert acts.shape[0] > 0


def test_does_not_process_the_whole_corpus_once_capped(gpt2):
    """The cap must actually short-circuit the loop, not just truncate the final result --
    otherwise a corpus far larger than max_tokens costs the same as one exactly max_tokens
    long, defeating the point of capping it."""
    model, tok = gpt2
    huge = TEXTS * 200  # far more than max_tokens could ever need
    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=huge, max_tokens=20,
                                      batch_size=4, max_length=32)
    assert acts.shape[0] <= 20 + 4 * 32  # at most one extra in-flight batch over the cap


def test_excludes_the_sink_by_default(gpt2):
    """Every intervention skips the sink (DECISIONS D4); the denoiser must never be trained to
    reconstruct it, since it will never be asked to at inference."""
    model, tok = gpt2
    from steering import hooks

    text = ["The capital of France is a city with a long history."]
    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook, torch.no_grad():
        enc = tok(text, return_tensors="pt")
        model(**enc)
    sink = hook.captured[0][0, 0, :]

    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=text, max_tokens=1000,
                                      batch_size=1, max_length=32, exclude_sink=True)
    # The sink's norm is ~30-40x the median (DEVLOG); if it leaked in, it would be the max.
    assert float(acts.norm(dim=-1).max()) < 0.5 * float(sink.norm())


def test_can_include_the_sink_when_asked(gpt2):
    model, tok = gpt2
    text = ["Hi."]
    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=text, max_tokens=1000,
                                      batch_size=1, max_length=32, exclude_sink=False)
    with_sink_max = float(acts.norm(dim=-1).max())

    acts_excl = denoiser.cache_activations(model, tok, layer=LAYER, texts=text, max_tokens=1000,
                                           batch_size=1, max_length=32, exclude_sink=True)
    without_sink_max = float(acts_excl.norm(dim=-1).max())
    assert with_sink_max > without_sink_max


def test_raises_on_empty_corpus(gpt2):
    model, tok = gpt2
    with pytest.raises(ValueError, match="no activations"):
        denoiser.cache_activations(model, tok, layer=LAYER, texts=[], max_tokens=100)


def test_restores_padding_side(gpt2):
    model, tok = gpt2
    tok.padding_side = "left"
    denoiser.cache_activations(model, tok, layer=LAYER, texts=TEXTS[:2], max_tokens=100,
                               batch_size=2, max_length=32)
    assert tok.padding_side == "left"


# --- not centered or scaled: an exact identity check against the raw hook capture -----------

def test_activations_are_raw_not_centered_or_scaled(gpt2):
    """The whole point of this module's design: cached activations must equal exactly what the
    hook hands any intervention at inference, so spaces.encode() inside D() is the only place
    normalisation happens -- never twice, never baked into stored data."""
    from steering import hooks

    model, tok = gpt2
    text = ["The weather today is unusually warm for this time of year."]

    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook, torch.no_grad():
        enc = tok(text, return_tensors="pt")
        model(**enc)
    raw_non_sink = hook.captured[0][0, 1:, :]

    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=text, max_tokens=1000,
                                      batch_size=1, max_length=32)
    # Same text, same model, same seed-free deterministic forward -- must match exactly.
    torch.testing.assert_close(acts, raw_non_sink)


# --- device consistency: same trap as compute_feature_stats/sae_concept_score ---------------

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available here")
def test_works_when_model_is_on_mps():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")

    acts = denoiser.cache_activations(model, tok, layer=LAYER, texts=TEXTS, max_tokens=200,
                                      batch_size=4, max_length=32)
    assert acts.shape[1] == 768
    assert acts.shape[0] > 0


# ==============================================================================================
# ResidualMLPDenoiser: D(x, s) = x + f(N(x), e(s))
# ==============================================================================================

D_MODEL = 8


def make_denoiser(activation_scale=90.0, center=True, seed=0):
    torch.manual_seed(seed)
    return denoiser.ResidualMLPDenoiser(d_model=D_MODEL, activation_scale=activation_scale,
                                        center=center)


# --- identity at init: the module's own opening claim ----------------------------------------

def test_identity_at_init_any_t():
    d = make_denoiser()
    x = torch.randn(16, D_MODEL) * 90.0
    for t in (0.0, 0.3, 0.7, 1.0):
        out = d(x, t=t)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=0)


def test_identity_at_init_with_a_batch_of_different_t():
    d = make_denoiser()
    x = torch.randn(16, D_MODEL) * 90.0
    t = torch.rand(16)
    out = d(x, t=t)
    torch.testing.assert_close(out, x, atol=1e-5, rtol=0)


def test_output_shape_matches_input():
    d = make_denoiser()
    x = torch.randn(4, D_MODEL) * 90.0
    assert d(x, t=0.5).shape == x.shape


def test_forward_returns_the_input_dtype_even_though_weights_are_float32():
    """This module's own weights are always float32 (never worth bf16 for a 20M-param MLP),
    but Gemma's residual stream is bf16 -- and MPS's matmul kernel hard-asserts on a
    float32/bf16 mix rather than upcasting the way CPU/CUDA would (this is exactly what killed
    a real training run before the fix). ``D(x, t)`` must accept and return whatever dtype
    ``x`` arrives in, transparently, so a caller composing this with a bf16 model's hidden
    states never needs to know this module computes internally in float32."""
    d = make_denoiser()
    x = torch.randn(4, D_MODEL) * 90.0
    out = d(x.bfloat16(), t=0.5)
    assert out.dtype == torch.bfloat16


# --- the un-scale trap: correction must be applied in RAW units, not normalised --------------

def test_correction_is_applied_in_raw_units_not_normalised(monkeypatch):
    """The exact bug this project's own conventions name: forgetting to multiply the MLP's
    (normalised-space) output by activation_scale before adding it to raw x produces a
    correction ~90x too small here and would still 'run' without error -- plausible-looking,
    silently wrong. Forces a known, nonzero correction and checks the applied delta scales
    with activation_scale, not independently of it."""
    d = make_denoiser(activation_scale=90.0)
    # Bypass training: directly set the output head to a known, nonzero linear map so the
    # correction in *normalised* space is exactly `x_encoded @ W^T` for a known W.
    with torch.no_grad():
        d.out.weight.copy_(torch.eye(*d.out.weight.shape) * 0.1)
        d.out.bias.zero_()

    x = torch.randn(4, D_MODEL) * 90.0
    with torch.no_grad():
        out = d(x, t=0.5)
    delta = out - x
    # delta must NOT be the same order of magnitude as if activation_scale had been skipped
    # (i.e. as if the raw normalised-space correction, ~O(0.1), had been added directly).
    assert float(delta.norm()) > 1.0, (
        "correction looks like it was added in normalised units, not raw -- "
        "the un-scale step is missing or wrong"
    )


def test_activation_scale_multiplies_a_scale_independent_correction_exactly(monkeypatch):
    """Isolates just the final un-scale multiply from the rest of the (nonlinear) network:
    zero fc1's weight so the correction becomes a fixed vector independent of `encoded` (hence
    independent of `scale`) before it ever reaches `out` -- everything downstream of a constant
    input is itself constant, regardless of fc2/out's own weights. Only then must
    delta = correction_constant * activation_scale scale EXACTLY linearly.

    A weaker, ratio-based claim without this isolation does not hold in general: `scale` enters
    the network twice (dividing the input via encode, multiplying the output via decode), and
    SiLU in between makes the whole map nonlinear, so halving the encode-scale does not simply
    double the raw-space output for an arbitrary (untrained) set of weights. An earlier version
    of this test asserted that weaker claim directly and it failed at ratio 1.59, not 2.0 -- the
    test's premise was wrong, not the un-scale step; this version tests only what is actually
    guaranteed."""
    d1 = make_denoiser(activation_scale=90.0, seed=1)
    d2 = make_denoiser(activation_scale=180.0, seed=1)
    with torch.no_grad():
        for d in (d1, d2):
            d.fc1.weight.zero_()
            d.fc1.bias.fill_(0.3)
            d.out.weight.copy_(torch.eye(*d.out.weight.shape) * 0.1)
            d.out.bias.zero_()

    x = torch.randn(4, D_MODEL) * 90.0
    delta1 = d1(x, t=0.5) - x
    delta2 = d2(x, t=0.5) - x
    torch.testing.assert_close(delta2, delta1 * 2.0, rtol=1e-4, atol=1e-4)


# --- conditioning actually does something -----------------------------------------------------

def test_t_changes_the_correction_once_trained():
    d = make_denoiser()
    with torch.no_grad():
        d.out.weight.copy_(torch.eye(*d.out.weight.shape) * 0.1)
        d.out.bias.zero_()
    x = torch.randn(4, D_MODEL) * 90.0
    out_low_t = d(x, t=0.0)
    out_high_t = d(x, t=1.0)
    assert not torch.allclose(out_low_t, out_high_t)


# --- loss --------------------------------------------------------------------------------------

def test_denoising_loss_hand_computed():
    target = torch.tensor([[6.0, 8.0]])  # norm 10
    pred = torch.tensor([[3.0, 4.0]])    # diff = (-3, -4), norm^2 = 25
    loss = denoiser.denoising_loss(pred, target, eps=0.0)
    assert float(loss) == pytest.approx(25.0 / 100.0)  # ||pred-target||^2 / ||target||^2


def test_denoising_loss_is_zero_for_a_perfect_prediction():
    x = torch.randn(8, D_MODEL) * 90.0
    assert float(denoiser.denoising_loss(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_denoising_loss_eps_avoids_division_by_zero():
    pred = torch.ones(1, D_MODEL)
    target = torch.zeros(1, D_MODEL)
    loss = denoiser.denoising_loss(pred, target, eps=1e-6)
    assert torch.isfinite(loss)


# --- save / load round trip --------------------------------------------------------------------

def test_save_load_round_trips_weights_and_config(tmp_path):
    d = make_denoiser(activation_scale=123.4, center=True, seed=5)
    with torch.no_grad():
        d.out.weight.copy_(torch.randn(*d.out.weight.shape) * 0.05)
        d.out.bias.copy_(torch.randn(D_MODEL) * 0.01)

    path = tmp_path / "denoiser.pt"
    denoiser.save_denoiser(d, path, extra={"corruption": {"corruption": "D3"}})
    loaded = denoiser.load_denoiser(path)

    x = torch.randn(4, D_MODEL) * 90.0
    torch.testing.assert_close(loaded(x, t=0.5), d(x, t=0.5))
    assert float(loaded.activation_scale) == pytest.approx(123.4)


def test_save_load_preserves_extra_metadata(tmp_path):
    d = make_denoiser()
    path = tmp_path / "denoiser.pt"
    denoiser.save_denoiser(d, path, extra={"corruption": {"corruption": "D4"}, "seed": 0})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["extra"]["corruption"]["corruption"] == "D4"


def test_config_is_json_safe():
    from steering.io import _json_safe

    d = make_denoiser(activation_scale=90.0)
    assert _json_safe(d.config)


# ==============================================================================================
# train_denoiser: the actual training loop
# ==============================================================================================

from steering import corruptions


def synthetic_pool(n=2000, d=D_MODEL, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * 90.0


def test_loss_decreases_substantially_over_training():
    """Not just 'runs without error' -- a real, if carefully bounded, threshold.

    `synthetic_pool` draws i.i.d. Gaussian rows with no structure a denoiser could exploit
    beyond simple shrinkage, so this problem has a computable Bayes-optimal floor: clean and
    noise are independent zero-mean Gaussians (var 90^2 and (sigma*90)^2), so the MMSE estimate
    is `x / (1+sigma^2)`, giving an *achievable* loss around 0.15 against an identity (untrained)
    loss around 0.216 -- measured directly, not derived on paper only. A first draft of this
    test demanded `final_loss < 0.5 * initial_loss` (~0.107), which is below the problem's own
    information-theoretic floor and therefore unreachable by any denoiser, trained or not; the
    training loop was already converging to near-optimal within ~150 steps when that assertion
    failed. Fixed to an achievable, still-meaningful bound instead of a smaller arbitrary one.
    """
    activations = synthetic_pool(n=4000)
    corruption = corruptions.Gaussian(sigma_min=0.1, sigma_max=1.0)  # modest, learnable range

    _model, history = denoiser.train_denoiser(
        activations, corruption, d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=300, batch_size=128, lr=3e-3, seed=0, device="cpu", log_every=50,
    )
    initial_loss = history[0]["loss"]
    final_loss = history[-1]["loss"]
    assert final_loss < 0.8 * initial_loss, (
        f"loss barely moved: {initial_loss:.4f} -> {final_loss:.4f}"
    )


def test_training_is_reproducible_with_the_same_seed():
    activations = synthetic_pool(n=1000)
    corruption = corruptions.Gaussian()
    _, history_a = denoiser.train_denoiser(
        activations, corruption, d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=50, batch_size=64, seed=0, device="cpu", log_every=10,
    )
    corruption2 = corruptions.Gaussian()  # fresh instance, no shared state
    _, history_b = denoiser.train_denoiser(
        activations, corruption2, d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=50, batch_size=64, seed=0, device="cpu", log_every=10,
    )
    assert [h["loss"] for h in history_a] == [h["loss"] for h in history_b]


def test_history_includes_identity_gap_on_genuinely_clean_data():
    activations = synthetic_pool(n=1000)
    corruption = corruptions.Gaussian()
    _, history = denoiser.train_denoiser(
        activations, corruption, d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=20, batch_size=64, seed=0, device="cpu", log_every=10,
    )
    assert all("identity_gap" in h for h in history)
    assert all(h["identity_gap"] >= 0 for h in history)


@pytest.mark.parametrize("make_corruption", [
    lambda: corruptions.Gaussian(),
    lambda: corruptions.VariancePreserving(),
    lambda: corruptions.Rank1(),
])
def test_train_denoiser_works_with_every_corruption_shape(make_corruption):
    """train_denoiser must not special-case any one corruption -- Corrupted's x/target/t
    interface is what makes D1/D2/D3/D4 interchangeable here."""
    activations = synthetic_pool(n=500)
    _model, history = denoiser.train_denoiser(
        activations, make_corruption(), d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=20, batch_size=32, seed=0, device="cpu", log_every=10,
    )
    assert all(math.isfinite(h["loss"]) for h in history)


def test_train_denoiser_works_with_structured_rank1():
    """D3/D4's Corrupted also carries `direction`, unlike D1/D2 -- must not break anything."""
    from steering.vectors import VectorSplit

    decoder = torch.randn(200, D_MODEL)
    split = VectorSplit(dev=list(range(20)), test=list(range(20, 60)), pool_size=200, seed=0,
                        model="test", source="test", created="")
    corruption = corruptions.FixedPoolRank1(decoder, split, pool_size=50)
    activations = synthetic_pool(n=500)
    _model, history = denoiser.train_denoiser(
        activations, corruption, d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=20, batch_size=32, seed=0, device="cpu", log_every=10,
    )
    assert all(math.isfinite(h["loss"]) for h in history)


# ==============================================================================================
# pool_memorization_curve: Step 9A -- does a fixed pool memorise its own directions?
# ==============================================================================================

def test_pool_memorization_curve_shows_more_overfitting_at_a_smaller_pool():
    """A tiny fixed pool should reconstruct directions it trained on (``own_loss``) far better
    than one it never saw (``unseen_loss``, ``dev_probe``) -- the memorisation signature Step 6
    measured at real GPT-2 scale (dev-generalization loss 0.168 at pool 64 vs 0.121 at the full
    pool, DEVLOG 2026-08-22: diversity helps up to ~1024 directions, then saturates). A much
    wider pool should show a smaller seen/unseen gap: with more directions to cover, "seen" and
    "unseen" are more similar distributions, leaving less room for genuine memorisation.
    """
    from steering.vectors import VectorSplit

    decoder = torch.randn(60, D_MODEL, generator=torch.Generator().manual_seed(0))
    split = VectorSplit(dev=list(range(10)), test=list(range(10, 20)), pool_size=60, seed=0,
                        model="test", source="test", created="")
    dev_probe = corruptions.StructuredRank1(decoder, pool_indices=split.dev, holdout=set(),
                                            rho_min=0.05, rho_max=1.0)
    activations = synthetic_pool(n=2000)

    rows = denoiser.pool_memorization_curve(
        activations, activations, decoder, split, dev_probe,
        pool_sizes=[2, None], d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=300, batch_size=64, lr=3e-3, seed=0, n_examples=1024, eval_seed=999, device="cpu",
    )

    assert [r["pool_size"] for r in rows] == [2, "full"]
    assert rows[0]["n_directions"] == 2
    assert rows[1]["n_directions"] == len(split.train_pool())
    assert all(math.isfinite(r["memorization_ratio"]) for r in rows)
    assert rows[0]["memorization_ratio"] > rows[1]["memorization_ratio"], (
        f"the 2-direction pool should overfit relatively harder than the full pool: "
        f"{rows[0]} vs {rows[1]}"
    )


def test_returned_model_is_in_eval_mode_with_grad_disabled():
    """A model handed back for evaluation must not still be accumulating gradients or have
    dropout/BN train-mode side effects -- this architecture has neither, but the convention
    (matching load_denoiser) should hold regardless."""
    activations = synthetic_pool(n=500)
    model, _ = denoiser.train_denoiser(
        activations, corruptions.Gaussian(), d_model=D_MODEL, activation_scale=90.0,
        center=False, steps=10, batch_size=32, seed=0, device="cpu", log_every=5,
    )
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


# --- device: cached activations are always CPU (ResidualHook's own design); training on MPS
# must move minibatches there without requiring the whole pool to live on-device -------------

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available here")
def test_train_denoiser_works_when_activations_are_cpu_but_device_is_mps():
    activations = synthetic_pool(n=1000)  # deliberately left on CPU
    model, history = denoiser.train_denoiser(
        activations, corruptions.Gaussian(), d_model=D_MODEL, activation_scale=90.0,
        center=False, steps=20, batch_size=64, seed=0, device="mps", log_every=10,
    )
    assert next(model.parameters()).device.type == "mps"
    assert all(math.isfinite(h["loss"]) for h in history)


# ==============================================================================================
# evaluate_denoiser: the function the "eliminate weak families" decision actually reads
# ==============================================================================================

def test_evaluate_returns_a_finite_float():
    activations = synthetic_pool(n=1000)
    model = make_denoiser()
    loss = denoiser.evaluate_denoiser(model, activations, corruptions.Gaussian(),
                                      n_examples=200, batch_size=64, seed=0, device="cpu")
    assert math.isfinite(loss)


def test_evaluate_is_reproducible_with_the_same_seed():
    activations = synthetic_pool(n=1000)
    model = make_denoiser()
    a = denoiser.evaluate_denoiser(model, activations, corruptions.Gaussian(),
                                   n_examples=200, batch_size=64, seed=1, device="cpu")
    b = denoiser.evaluate_denoiser(model, activations, corruptions.Gaussian(),
                                   n_examples=200, batch_size=64, seed=1, device="cpu")
    assert a == b


def test_evaluate_differs_with_a_different_seed():
    activations = synthetic_pool(n=1000)
    model = make_denoiser()
    a = denoiser.evaluate_denoiser(model, activations, corruptions.Gaussian(),
                                   n_examples=200, batch_size=64, seed=1, device="cpu")
    b = denoiser.evaluate_denoiser(model, activations, corruptions.Gaussian(),
                                   n_examples=200, batch_size=64, seed=2, device="cpu")
    assert a != b


def test_a_trained_model_beats_an_untrained_one_on_held_out_data():
    """The actual point of this function: it must measure real denoising quality, not just
    return a number. An untrained (identity) model and a model trained on the easy synthetic
    problem must be distinguishable on data neither saw during training."""
    train_pool = synthetic_pool(n=3000, seed=0)
    held_out = synthetic_pool(n=500, seed=99)  # disjoint draw, different seed
    corruption = corruptions.Gaussian(sigma_min=0.1, sigma_max=1.0)

    trained, _ = denoiser.train_denoiser(
        train_pool, corruption, d_model=D_MODEL, activation_scale=90.0, center=False,
        steps=300, batch_size=128, lr=3e-3, seed=0, device="cpu", log_every=300,
    )
    untrained = make_denoiser(seed=1)  # zero-init -> exactly identity

    loss_trained = denoiser.evaluate_denoiser(trained, held_out, corruption, n_examples=500,
                                              seed=42, device="cpu")
    loss_untrained = denoiser.evaluate_denoiser(untrained, held_out, corruption, n_examples=500,
                                                seed=42, device="cpu")
    assert loss_trained < loss_untrained


@pytest.mark.parametrize("make_corruption", [
    lambda: corruptions.Gaussian(),
    lambda: corruptions.VariancePreserving(),
    lambda: corruptions.Rank1(),
])
def test_evaluate_works_with_every_corruption_shape(make_corruption):
    activations = synthetic_pool(n=500)
    model = make_denoiser()
    loss = denoiser.evaluate_denoiser(model, activations, make_corruption(), n_examples=100,
                                      batch_size=32, seed=0, device="cpu")
    assert math.isfinite(loss)


def test_evaluate_handles_n_examples_not_a_multiple_of_batch_size():
    activations = synthetic_pool(n=500)
    model = make_denoiser()
    loss = denoiser.evaluate_denoiser(model, activations, corruptions.Gaussian(),
                                      n_examples=137, batch_size=64, seed=0, device="cpu")
    assert math.isfinite(loss)

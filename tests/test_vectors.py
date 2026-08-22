"""Vector sources (SAE decoder rows, persona directions) and the frozen holdout split.

The holdout is the one property in this whole project that must never be wrong: if a
corruption sampler ever draws a DEV or TEST direction, every number computed afterward is
invalid, silently. ``VectorSplit`` makes this structural rather than merely checked: TRAIN is
defined as the complement of DEV/TEST over the pool, so it cannot drift out of sync with the
holdout the way a separately-maintained list could.

SAE-side tests run against the real GPT-2 SAE (``gpt2-small-resid-post-v5-128k``,
``blocks.6.hook_resid_post``, TopK-32, d_sae=131072) -- offline and already cached -- because
the guards here exist specifically to catch this SAE's own failure modes (the resid_pre/
resid_post off-by-one, MPS returning corrupt encodes under memory pressure).
"""

from __future__ import annotations

import json

import pytest
import torch

from steering import vectors

LAYER = 6
RELEASE, SAE_ID = "gpt2-small-resid-post-v5-128k", "blocks.6.hook_resid_post"


@pytest.fixture(scope="module")
def sae():
    return vectors.load_sae(RELEASE, SAE_ID, layer=LAYER, d_model=768, device="cpu")


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


# --- SAE loading and its correctness guards -------------------------------------------------

def test_load_sae_reports_expected_shapes(sae):
    assert sae.cfg.d_in == 768
    assert sae.cfg.d_sae == 131072


def test_load_sae_wrong_d_model_raises():
    with pytest.raises(ValueError, match="d_in"):
        vectors.load_sae(RELEASE, SAE_ID, layer=LAYER, d_model=2304, device="cpu")


def test_load_sae_layer_mismatch_raises():
    """The resid_pre/resid_post off-by-one, guarded: an SAE trained on a different block's
    hook must not silently load as if it matched this project's intervention layer."""
    with pytest.raises(ValueError, match="layer"):
        vectors.load_sae(RELEASE, SAE_ID, layer=5, d_model=768, device="cpu")


def test_layer_match_accepts_the_resid_pre_naming_of_the_next_block():
    """resid_post(6) == resid_pre(7): the SAME activation, two hook-name conventions. An SAE
    whose metadata says ``blocks.7.hook_resid_pre`` is therefore also a correct match for
    intervention layer=6 -- not a different layer, the same point named the other way.
    """
    class FakeMeta:
        hook_name = "blocks.7.hook_resid_pre"

    class FakeCfg:
        metadata = FakeMeta()

    class FakeSAE:
        cfg = FakeCfg()

    vectors._assert_layer_matches(FakeSAE(), layer=6)  # must not raise


def test_layer_match_rejects_the_true_neighbouring_layer():
    """The failure this whole guard exists for: layer=6's SAE used at layer=7 is a different,
    adjacent activation -- plausible-looking, wrong. Confirmed against the real SAE, since this
    is the exact off-by-one the exploratory repo names as a correctness trap."""
    with pytest.raises(ValueError, match="layer"):
        vectors.load_sae(RELEASE, SAE_ID, layer=7, d_model=768, device="cpu")


# --- sae_encode and its sparsity guard --------------------------------------------------------

def test_encode_respects_topk_invariant(sae):
    x = torch.randn(20, 768) * 90  # roughly GPT-2 layer-6 scale
    features = vectors.sae_encode(sae, x)
    l0 = (features > 0).sum(-1)
    assert int(l0.max()) == sae.cfg.k
    assert int(l0.min()) == sae.cfg.k


def test_encode_chunking_matches_a_single_call(sae):
    x = torch.randn(37, 768) * 90
    whole = vectors.sae_encode(sae, x, chunk=1000)
    chunked = vectors.sae_encode(sae, x, chunk=8)
    torch.testing.assert_close(whole, chunked)


def test_encode_raises_on_a_violated_topk_invariant():
    """The guard, not the SAE: a fake SAE whose encode ignores its own k must be caught, since
    this is exactly what corrupt MPS output looks like (measured in the exploratory repo: FVU
    23.2, L0 16 under a TopK-32 SAE, no exception)."""

    class FakeCfg:
        k = 32

    class FakeSAE:
        cfg = FakeCfg()

        def encode(self, x):
            return torch.ones(x.shape[0], 100)  # every feature "fires" -- l0=100 > k=32

    with pytest.raises(RuntimeError, match="TopK invariant violated"):
        vectors.sae_encode(FakeSAE(), torch.randn(4, 768))


# --- steering_directions ----------------------------------------------------------------------

def test_directions_are_unit_norm_by_default(sae):
    d = vectors.steering_directions(sae, [10, 20, 30])
    torch.testing.assert_close(d.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=0)


def test_directions_unnormalized_preserve_decoder_scale(sae):
    d = vectors.steering_directions(sae, [10, 20, 30], normalize=False)
    torch.testing.assert_close(d, sae.W_dec[torch.tensor([10, 20, 30])])


def test_directions_shape_matches_request(sae):
    d = vectors.steering_directions(sae, [1, 2, 3, 4, 5])
    assert d.shape == (5, 768)


# --- feature stats over a corpus --------------------------------------------------------------

@pytest.fixture(scope="module")
def texts():
    return [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "In 1969, humans first set foot on the surface of the Moon.",
        "She opened the door slowly, unsure of what she would find inside.",
        "Markets fell sharply today amid concerns over interest rate policy.",
        "The recipe calls for two cups of flour and a pinch of salt.",
        "He studied the ancient manuscript for clues about its origin.",
    ] * 4


def test_feature_stats_frequency_is_a_valid_probability(gpt2, sae, texts):
    model, tok = gpt2
    stats = vectors.compute_feature_stats(
        model, tok, sae, layer=LAYER, texts=texts, batch_size=4, max_length=32, center=True
    )
    assert stats["frequency"].shape == (sae.cfg.d_sae,)
    assert bool((stats["frequency"] >= 0).all()) and bool((stats["frequency"] <= 1).all())


def test_feature_stats_mean_l0_matches_k_for_topk(gpt2, sae, texts):
    """A TopK SAE fires exactly k features per token, always -- this is an identity, not a
    measurement, and a wrong value here means the stats loop is double-counting or masking
    incorrectly."""
    model, tok = gpt2
    stats = vectors.compute_feature_stats(
        model, tok, sae, layer=LAYER, texts=texts, batch_size=4, max_length=32, center=True
    )
    assert float(stats["mean_l0"]) == pytest.approx(sae.cfg.k, abs=0.01)


def test_feature_stats_raises_on_empty_corpus(gpt2, sae):
    model, tok = gpt2
    with pytest.raises(ValueError, match="no tokens"):
        vectors.compute_feature_stats(model, tok, sae, layer=LAYER, texts=[], center=True)


def test_feature_stats_restores_padding_side(gpt2, sae, texts):
    """generate.py needs left-padding; this function needs right-padding for its sink
    assumption to hold. Neither may leave the tokenizer's padding_side changed for the other,
    since nothing in this project sets a global convention once for every caller."""
    model, tok = gpt2
    tok.padding_side = "left"
    vectors.compute_feature_stats(model, tok, sae, layer=LAYER, texts=texts, batch_size=4,
                                  max_length=32, center=True)
    assert tok.padding_side == "left"


def test_feature_stats_is_correct_under_left_padding_too(gpt2, sae):
    """The actual bug this guards: with left-padding, a short sequence's position 0 is a pad
    token, not its sink. If the function trusted the caller's padding_side instead of forcing
    its own, this would silently exclude the wrong token for the shorter sequence."""
    model, tok = gpt2
    short, long_ = "The cat sat.", "The quick brown fox jumps over the lazy dog today"
    tok.padding_side = "left"
    stats_left_caller = vectors.compute_feature_stats(
        model, tok, sae, layer=LAYER, texts=[short, long_], batch_size=2,
        max_length=16, center=True,
    )
    tok.padding_side = "right"
    stats_right_caller = vectors.compute_feature_stats(
        model, tok, sae, layer=LAYER, texts=[short, long_], batch_size=2,
        max_length=16, center=True,
    )
    torch.testing.assert_close(stats_left_caller["frequency"], stats_right_caller["frequency"])


# --- frequency band ----------------------------------------------------------------------

def test_frequency_band_selects_the_middle():
    frequency = torch.tensor([0.0, 1e-5, 1e-4, 5e-4, 5e-3, 1e-2, 0.5])
    stats = {"frequency": frequency}
    idx = vectors.frequency_band(stats, freq_min=1e-4, freq_max=5e-3)
    assert idx.tolist() == [2, 3, 4]


def test_band_from_mean_scales_with_the_measured_mean():
    frequency = torch.full((1000,), 2.44e-4)
    stats = {"frequency": frequency}
    lo, hi = vectors.band_from_mean(stats, lo=0.41, hi=20.5)
    assert lo == pytest.approx(0.41 * 2.44e-4, rel=1e-6)
    assert hi == pytest.approx(20.5 * 2.44e-4, rel=1e-6)


# --- VectorSplit: the holdout ------------------------------------------------------------------

def make_split(**overrides):
    kwargs = {
        "dev": [1, 2, 3], "test": [10, 11, 12], "pool_size": 100, "seed": 0,
        "model": "gpt2", "source": "gpt2_sae_blocks.6.hook_resid_post",
        "selection": {"method": "test"}, "created": "2026-08-22T00:00:00+00:00",
    }
    kwargs.update(overrides)
    return vectors.VectorSplit(**kwargs)


def test_disjoint_dev_test_construct_cleanly():
    split = make_split()
    assert split.dev == [1, 2, 3]
    assert split.test == [10, 11, 12]


def test_overlapping_dev_test_raises():
    with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
        make_split(dev=[1, 2, 3], test=[3, 4, 5])


def test_out_of_range_index_raises():
    with pytest.raises(ValueError, match="pool_size"):
        make_split(pool_size=10, test=[10, 11, 12])


def test_train_pool_is_the_structural_complement():
    """TRAIN is derived, never stored -- so it cannot go stale relative to a hand-edited
    dev/test list the way a separately-maintained list could."""
    split = make_split(pool_size=10, dev=[1, 2], test=[8, 9])
    assert split.train_pool() == [0, 3, 4, 5, 6, 7]


def test_train_pool_never_intersects_holdout_by_construction():
    split = make_split(pool_size=200, dev=list(range(20)), test=list(range(180, 200)))
    assert set(split.train_pool()) & split.holdout == set()
    assert len(split.train_pool()) + len(split.holdout) == 200


def test_fingerprint_is_stable_across_reconstruction():
    a, b = make_split(), make_split()
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_if_a_single_test_id_changes():
    a = make_split()
    b = make_split(test=[10, 11, 13])  # one id different
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_ignores_list_order():
    a = make_split(dev=[3, 1, 2], test=[12, 10, 11])
    b = make_split(dev=[1, 2, 3], test=[10, 11, 12])
    assert a.fingerprint() == b.fingerprint()


def test_save_and_load_round_trips(tmp_path):
    split = make_split()
    path = tmp_path / "split.json"
    vectors.save_split(split, path)
    loaded = vectors.load_split(path)
    assert loaded.dev == split.dev
    assert loaded.test == split.test
    assert loaded.train_pool() == split.train_pool()


def test_load_detects_a_hand_edited_test_set(tmp_path):
    """The failure this exists for: someone edits the frozen split file by hand -- adding a
    concept back in after seeing it looked bad, say -- and every later run silently uses it."""
    split = make_split()
    path = tmp_path / "split.json"
    vectors.save_split(split, path)

    payload = json.loads(path.read_text())
    payload["test"] = [10, 11, 13]  # tampered (13, not 12), fingerprint now stale
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
        vectors.load_split(path)


# --- select_and_freeze_split --------------------------------------------------------------

def test_select_frequency_only_produces_disjoint_dev_test(tmp_path):
    frequency = torch.zeros(1000)
    frequency[100:400] = 1e-3  # 300 in-band candidates
    stats = {"frequency": frequency, "n_tokens": torch.tensor(50_000)}
    split = vectors.select_and_freeze_split(
        stats, tmp_path / "split.json", freq_min=1e-4, freq_max=5e-3,
        n_dev=20, n_test=50, seed=0, model="gpt2", source="gpt2_sae_test",
    )
    assert len(split.dev) == 20
    assert len(split.test) == 50
    assert split.holdout.issubset(set(range(100, 400)))
    assert split.pool_size == 1000


def test_select_too_few_candidates_raises(tmp_path):
    frequency = torch.zeros(1000)
    frequency[100:110] = 1e-3  # only 10 in-band
    stats = {"frequency": frequency, "n_tokens": torch.tensor(1000)}
    with pytest.raises(ValueError, match="[Oo]nly"):
        vectors.select_and_freeze_split(
            stats, tmp_path / "split.json", freq_min=1e-4, freq_max=5e-3,
            n_dev=20, n_test=50, seed=0, model="gpt2", source="gpt2_sae_test",
        )


def test_select_with_steerability_filters_unusable_features(tmp_path):
    frequency = torch.zeros(1000)
    frequency[0:200] = 1e-3
    stats = {"frequency": frequency, "n_tokens": torch.tensor(50_000)}
    # Only even-numbered features in-band are "usable".
    steerability = {f: {"peak": 0.05, "usable": f % 2 == 0} for f in range(200)}
    split = vectors.select_and_freeze_split(
        stats, tmp_path / "split.json", freq_min=1e-4, freq_max=5e-3,
        n_dev=10, n_test=20, seed=0, model="gpt2", source="gpt2_sae_test",
        steerability=steerability,
    )
    assert all(f % 2 == 0 for f in split.holdout)
    assert split.selection["method"] != "frequency_band_only"


def test_select_is_reproducible_given_the_same_seed(tmp_path):
    frequency = torch.zeros(1000)
    frequency[0:300] = 1e-3
    stats = {"frequency": frequency, "n_tokens": torch.tensor(50_000)}
    a = vectors.select_and_freeze_split(
        stats, tmp_path / "a.json", freq_min=1e-4, freq_max=5e-3,
        n_dev=10, n_test=20, seed=7, model="gpt2", source="s",
    )
    b = vectors.select_and_freeze_split(
        stats, tmp_path / "b.json", freq_min=1e-4, freq_max=5e-3,
        n_dev=10, n_test=20, seed=7, model="gpt2", source="s",
    )
    assert a.dev == b.dev
    assert a.test == b.test


# --- persona vectors: chat-template handling ---------------------------------------------

class _FakeTokenizerNoTemplate:
    def apply_chat_template(self, *a, **k):
        raise ValueError("no chat template")


class _FakeTokenizerGemmaLike:
    """Has a template, but raises on a system turn -- Gemma's actual behaviour."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + (
            "<assistant>" if add_generation_prompt else ""
        )


class _FakeTokenizerFullChat:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + (
            "<assistant>" if add_generation_prompt else ""
        )


def test_has_chat_template_false_for_gpt2_like():
    assert vectors.has_chat_template(_FakeTokenizerNoTemplate()) is False


def test_has_chat_template_true_for_gemma_like():
    assert vectors.has_chat_template(_FakeTokenizerGemmaLike()) is True


def test_supports_system_role_false_for_gemma_like():
    """The specific fact that makes Gemma special: it has a template but no system turn."""
    assert vectors.supports_system_role(_FakeTokenizerGemmaLike()) is False


def test_supports_system_role_true_for_full_chat():
    assert vectors.supports_system_role(_FakeTokenizerFullChat()) is True


def test_build_prompt_folds_instruction_into_user_turn_on_gemma_like():
    prompt = vectors.build_prompt(_FakeTokenizerGemmaLike(), "Be evil.", "How are you?")
    assert "<system>" not in prompt
    assert "Be evil." in prompt and "How are you?" in prompt


def test_build_prompt_uses_system_turn_when_supported():
    prompt = vectors.build_prompt(_FakeTokenizerFullChat(), "Be evil.", "How are you?")
    assert prompt.startswith("<system>Be evil.")


def test_build_prompt_falls_back_to_plain_concat_without_a_template():
    prompt = vectors.build_prompt(_FakeTokenizerNoTemplate(), "Be evil.", "How are you?")
    assert "Be evil." in prompt and "How are you?" in prompt


# --- persona vectors: Trait and the extraction/eval question split -----------------------

def test_trait_splits_extraction_and_eval_questions_without_overlap():
    trait = vectors.Trait(
        name="evil", description="d", positive=["p"], negative=["n"],
        questions=[f"q{i}" for i in range(30)],
    )
    extract, eval_ = trait.split_questions(20)
    assert len(extract) == 20 and len(eval_) == 10
    assert set(extract) & set(eval_) == set()


def test_cohens_d_is_zero_for_identical_distributions():
    x = torch.randn(50)
    assert vectors.cohens_d(x, x.clone()) == pytest.approx(0.0, abs=1e-6)


def test_cohens_d_is_positive_when_a_exceeds_b():
    a = torch.full((50,), 5.0) + torch.randn(50) * 0.1
    b = torch.full((50,), 0.0) + torch.randn(50) * 0.1
    assert vectors.cohens_d(a, b) > 5.0


# --- persona vectors: extraction, end to end on a real (non-chat) model ------------------

def test_extract_persona_vector_end_to_end(gpt2):
    """GPT-2 has no chat template, exercising the plain-concatenation path with a real model
    and a real hook -- the mechanics under test here (masking, averaging, direction, Cohen's
    d), not the semantic quality of a GPT-2 'trait', which this MVP does not claim."""
    model, tok = gpt2
    trait = vectors.Trait(
        name="formal",
        description="speaks formally",
        positive=["Respond in a very formal, professional register."],
        negative=["Respond in a very casual, sloppy register."],
        questions=["Tell me about your day.", "What do you think of the weather?"],
    )
    result = vectors.extract_persona_vector(
        model, tok, trait, layer=LAYER, device="cpu", n_extract=2, max_new_tokens=8, seed=0,
    )
    vector = torch.tensor(result["vector"])
    assert vector.shape == (768,)
    assert vector.norm() == pytest.approx(1.0, rel=1e-4)
    assert result["trait"] == "formal"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available here")
def test_extract_persona_vector_works_when_model_is_on_mps():
    """Same device trap as compute_feature_stats/sae_concept_score: answer_activations also
    captures via ResidualHook(capture=True), which is always CPU, while the answer-mask it
    combines it with is built from `generated`, on `device`. Relevant for real Step 11 Gemma
    runs, not just this GPT-2 smoke test."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")
    trait = vectors.Trait(
        name="formal", description="speaks formally",
        positive=["Respond in a very formal, professional register."],
        negative=["Respond in a very casual, sloppy register."],
        questions=["Tell me about your day.", "What do you think of the weather?"],
    )
    result = vectors.extract_persona_vector(
        model, tok, trait, layer=LAYER, device="mps", n_extract=2, max_new_tokens=8, seed=0,
    )
    assert torch.tensor(result["vector"]).shape == (768,)


# --- probe_steerability: the dual fluency guard ------------------------------------------

class _FakeSAEForProbe:
    """Only W_dec is touched before the monkeypatched generate/metrics take over."""

    W_dec = torch.randn(64, 8)


def test_probe_selects_the_peak_among_fluent_r_only():
    """The rule (DECISIONS D1): peak concept gain among r values passing BOTH ppl_ratio<=4 and
    dist_2>=0.8*baseline. Every generate/metrics call is monkeypatched to return controlled,
    hand-picked values, so this pins the selection arithmetic exactly rather than depending on
    real model behaviour to happen to exercise every branch."""
    import types

    from steering import generate as generate_module
    from steering import metrics as metrics_module

    # r=0.25 fluent, small gain; r=0.5 fluent, BEST gain; r=1.0 fails ppl_ratio despite a huge
    # apparent gain (must be excluded); r=2.0 fails dist_2 despite passing ppl_ratio.
    PLAN = {
        0.25: {"ppl_ratio": 1.2, "dist2": 0.95, "gain": 0.05},
        0.5: {"ppl_ratio": 2.0, "dist2": 0.85, "gain": 0.20},
        1.0: {"ppl_ratio": 5.0, "dist2": 0.90, "gain": 0.90},  # excluded: ppl too high
        2.0: {"ppl_ratio": 1.5, "dist2": 0.50, "gain": 0.99},  # excluded: dist2 too low
    }
    BASELINE_DIST2 = 1.0
    BASELINE_ACT = 0.01

    def fake_generate(model, tokenizer, prompts, intervention, layer, device, **kw):
        r = getattr(intervention, "r", 0.0)
        return types.SimpleNamespace(texts=[f"r={r}"] * len(prompts), n_decode_steps=[1])

    def fake_reference_nll(model, tokenizer, prompts, texts, device, **kw):
        import torch as _t

        r = float(texts[0].split("=")[1]) if texts and texts[0] != "r=0.0" else 0.0
        ratio = PLAN.get(r, {"ppl_ratio": 1.0})["ppl_ratio"]
        # reference_nll's caller does .exp() on the mean; encode ratio via log so the caller's
        # arithmetic (exp(mean)) reproduces the intended ppl_ratio relative to a baseline of
        # ppl=1 (log(1)=0), independent of the real baseline_ppl computed from r=0.
        import math

        return _t.tensor([math.log(ratio)] * len(prompts))

    def fake_distinct_n(texts, n):
        r = float(texts[0].split("=")[1]) if texts and texts[0] != "r=0.0" else 0.0
        return PLAN.get(r, {"dist2": BASELINE_DIST2})["dist2"] * BASELINE_DIST2

    def fake_concept_score(model, tokenizer, sae, feature_id, layer, prompts, continuations,
                           device, center, **kw):
        r = float(continuations[0].split("=")[1]) if continuations[0] != "r=0.0" else 0.0
        if r == 0.0:
            return {"mean_act": BASELINE_ACT, "fire_rate": 0.01}
        return {"mean_act": BASELINE_ACT + PLAN[r]["gain"], "fire_rate": 0.5}

    import steering.vectors as vectors_module

    orig = {
        "generate": generate_module.generate,
        "reference_nll": metrics_module.reference_nll,
        "distinct_n": metrics_module.distinct_n,
        "sae_concept_score": metrics_module.sae_concept_score,
    }
    generate_module.generate = fake_generate
    metrics_module.reference_nll = fake_reference_nll
    metrics_module.distinct_n = fake_distinct_n
    metrics_module.sae_concept_score = fake_concept_score
    try:
        result = vectors_module.probe_steerability(
            model=None, tokenizer=None, sae=_FakeSAEForProbe(), layer=6, feature_ids=[42],
            prompts=["p1", "p2"], scale=90.0, device="cpu", center=True,
            r_values=(0.25, 0.5, 1.0, 2.0),
        )
    finally:
        generate_module.generate = orig["generate"]
        metrics_module.reference_nll = orig["reference_nll"]
        metrics_module.distinct_n = orig["distinct_n"]
        metrics_module.sae_concept_score = orig["sae_concept_score"]

    r = result[42]
    assert r["r_at_peak"] == pytest.approx(0.5)
    assert r["peak"] == pytest.approx(PLAN[0.5]["gain"], rel=1e-4)
    assert r["n_r_fluent"] == 2  # only 0.25 and 0.5 pass both guards
    assert r["usable"] is True  # peak 0.20 > RESPONSIVE_THRESHOLD 0.02


def test_probe_marks_unusable_when_no_fluent_r_clears_the_gain_threshold():
    import types

    from steering import generate as generate_module
    from steering import metrics as metrics_module
    from steering import vectors as vectors_module

    def fake_generate(model, tokenizer, prompts, intervention, layer, device, **kw):
        r = getattr(intervention, "r", 0.0)
        return types.SimpleNamespace(texts=[f"r={r}"] * len(prompts), n_decode_steps=[1])

    def fake_reference_nll(model, tokenizer, prompts, texts, device, **kw):
        import torch as _t

        return _t.tensor([0.0] * len(prompts))  # ppl_ratio always 1.0 -- always fluent

    def fake_distinct_n(texts, n):
        return 1.0  # always at baseline -- always fluent

    def fake_concept_score(model, tokenizer, sae, feature_id, layer, prompts, continuations,
                           device, center, **kw):
        return {"mean_act": 0.01, "fire_rate": 0.01}  # never moves from baseline

    orig = {
        "generate": generate_module.generate,
        "reference_nll": metrics_module.reference_nll,
        "distinct_n": metrics_module.distinct_n,
        "sae_concept_score": metrics_module.sae_concept_score,
    }
    generate_module.generate = fake_generate
    metrics_module.reference_nll = fake_reference_nll
    metrics_module.distinct_n = fake_distinct_n
    metrics_module.sae_concept_score = fake_concept_score
    try:
        result = vectors_module.probe_steerability(
            model=None, tokenizer=None, sae=_FakeSAEForProbe(), layer=6, feature_ids=[7],
            prompts=["p1"], scale=90.0, device="cpu", center=True, r_values=(0.5, 1.0),
        )
    finally:
        generate_module.generate = orig["generate"]
        metrics_module.reference_nll = orig["reference_nll"]
        metrics_module.distinct_n = orig["distinct_n"]
        metrics_module.sae_concept_score = orig["sae_concept_score"]

    assert result[7]["usable"] is False
    assert result[7]["peak"] == pytest.approx(0.0)


# --- probe_steerability: small real integration ----------------------------------------------

def test_probe_runs_end_to_end_on_real_gpt2(gpt2, sae):
    model, tok = gpt2
    result = vectors.probe_steerability(
        model, tok, sae, layer=LAYER, feature_ids=[1878, 15452],
        prompts=["The best thing about", "Yesterday I went to"], scale=90.0, device="cpu",
        center=True, r_values=(0.5, 1.0), max_new_tokens=8,
    )
    assert set(result) == {1878, 15452}
    for record in result.values():
        assert "usable" in record and isinstance(record["usable"], bool)
        assert "baseline_act" in record


# --- device consistency: hidden captured to CPU vs. mask/model on MPS --------------------
# The bug this section exists for: ResidualHook(capture=True) always captures to CPU
# (deliberate -- caching a large corpus on MPS runs it out of memory), but a mask built from
# `enc`/`generated` lives on whatever device the model is on. Every test above ran model and
# SAE on CPU only, so hidden and mask always coincidentally matched -- this was invisible until
# a real MPS run surfaced it. Skipped, not failed, where MPS isn't available (e.g. CI, Linux).

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available here")
def test_compute_feature_stats_works_when_model_is_on_mps(sae):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")

    stats = vectors.compute_feature_stats(
        model, tok, sae, layer=LAYER,
        texts=["The quick brown fox jumps over the lazy dog.", "Yesterday I went to the market."],
        batch_size=2, max_length=32, center=True,
    )
    assert stats["frequency"].shape == (sae.cfg.d_sae,)
    assert float(stats["mean_l0"]) == pytest.approx(sae.cfg.k, abs=0.5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available here")
def test_compute_feature_stats_works_when_sae_is_on_mps():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to("mps")
    sae_mps = vectors.load_sae(RELEASE, SAE_ID, layer=LAYER, d_model=768, device="mps")

    stats = vectors.compute_feature_stats(
        model, tok, sae_mps, layer=LAYER,
        texts=["The quick brown fox jumps over the lazy dog."],
        batch_size=1, max_length=32, center=True,
    )
    assert stats["frequency"].shape == (sae_mps.cfg.d_sae,)


# --- Neuronpedia: token-level classification and cached fetch ---------------------------------

@pytest.mark.parametrize("description", [
    "the comma and other punctuation marks",
    "instances of the indefinite article 'a'",
    "articles and indefinite articles",  # plural -- the exact bug the trailing s? fixes
    "the word \"the\"",
    "line breaks and newline characters",
    "a single capital letter at the start of a word",
])
def test_token_level_descriptions_are_flagged(description):
    assert vectors.is_token_level(description) is True


@pytest.mark.parametrize("description", [
    "references to the Apple brand and its products",
    "phrases expressing pain and suffering",
    "athletic achievements, particularly records in running",
    "references to political candidates and elections",
])
def test_semantic_descriptions_are_not_flagged(description):
    assert vectors.is_token_level(description) is False


def test_fetch_feature_uses_the_cache_without_a_network_call(tmp_path, monkeypatch):
    cache_dir = tmp_path / "np_cache"
    path = cache_dir / "6-res_post_128k-oai" / "1878.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"explanations": [{"description": "cached description"}]}')

    def fail_if_called(*a, **kw):
        raise AssertionError("should not hit the network when the cache already has this file")

    monkeypatch.setattr(vectors.requests, "get", fail_if_called)
    payload = vectors.fetch_feature("gpt2-small/6-res_post_128k-oai", 1878, cache_dir)
    assert payload["explanations"][0]["description"] == "cached description"


def test_fetch_feature_writes_the_cache_on_a_network_hit(tmp_path, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"explanations": [{"description": "fetched live"}]}

    monkeypatch.setattr(vectors.requests, "get", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr(vectors.time, "sleep", lambda *a: None)

    cache_dir = tmp_path / "np_cache"
    payload = vectors.fetch_feature("gpt2-small/6-res_post_128k-oai", 42, cache_dir)
    assert payload["explanations"][0]["description"] == "fetched live"
    cached = (cache_dir / "6-res_post_128k-oai" / "42.json").read_text()
    assert "fetched live" in cached


def test_fetch_feature_returns_none_on_a_network_failure(tmp_path, monkeypatch):
    def raise_error(*a, **kw):
        raise ConnectionError("no network")

    monkeypatch.setattr(vectors.requests, "get", raise_error)
    assert vectors.fetch_feature("gpt2-small/x", 1, tmp_path / "np_cache") is None


def test_describe_features_omits_undescribed_features(tmp_path, monkeypatch):
    cache_dir = tmp_path / "np_cache"

    def fake_get(*a, **kw):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"explanations": []}  # no description available

        return Resp()

    monkeypatch.setattr(vectors.requests, "get", fake_get)
    monkeypatch.setattr(vectors.time, "sleep", lambda *a: None)
    out = vectors.describe_features("gpt2-small/6-res_post_128k-oai", [1878], cache_dir)
    assert out == {}


def test_describe_features_classifies_token_level(tmp_path, monkeypatch):
    cache_dir = tmp_path / "np_cache"
    responses = {
        1: "references to political candidates",
        2: "the comma and other punctuation",
    }

    def fake_get(url, timeout=25):
        index = int(url.rstrip("/").rsplit("/", 1)[-1])

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"explanations": [{"description": responses[index]}]}

        return Resp()

    monkeypatch.setattr(vectors.requests, "get", fake_get)
    monkeypatch.setattr(vectors.time, "sleep", lambda *a: None)
    out = vectors.describe_features("gpt2-small/6-res_post_128k-oai", [1, 2], cache_dir)
    assert out[1]["token_level"] is False
    assert out[2]["token_level"] is True


def test_save_and_load_persona_vectors_round_trip(tmp_path):
    payload = {"evil": {"trait": "evil", "vector": [0.1, 0.2, 0.3], "layer": 6}}
    path = tmp_path / "persona.json"
    vectors.save_persona_vectors(payload, path)
    loaded = vectors.load_persona_vectors(path)
    assert loaded["evil"].tolist() == pytest.approx([0.1, 0.2, 0.3])

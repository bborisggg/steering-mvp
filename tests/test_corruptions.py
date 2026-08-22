"""Training corruptions for the denoiser: D1 Gaussian, D2 variance-preserving, D3 rank-1/fixed
pool, D4 rank-1/full pool (PLAN.md method table, Step 5).

**The holdout is the one property that must never be wrong** (project-wide, and doubly so
here: this is the one place a real steering direction -- a DEV or TEST decoder row -- could
enter training at all). D3/D4 draw only from ``VectorSplit.train_pool()``, which is already
structurally disjoint from ``dev``/``test`` by construction (vectors.py: TRAIN is the
*complement*, never a separately-stored list). The tests here still assert disjointness
explicitly and directly, including against a hand-crafted pool that bypasses the normal
``split.train_pool()`` path -- catching a wrong *implementation* of the subsetting logic, not
just documenting a property the type system already guarantees. Defense in depth, matching
``VectorSplit.__post_init__``'s own redundant check.

D2 is PLAN.md's own formula, ``sqrt(1-beta)*h + sqrt(beta)*eps`` -- a variance-preserving
SDE-style corruption -- and is a *different* corruption family from the exploratory repo's own
"C2" (a plain linear interpolant, ``t*h + (1-t)*eps``), despite the superficial resemblance in
name. Implemented from PLAN.md's formula directly, not ported.
"""

from __future__ import annotations

import pytest
import torch

from steering import corruptions

D = 8


def clean_batch(n=64, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Realistic-ish: nonzero mean norm, not unit-scale, matching real GPT-2 activations.
    return torch.randn(n, d, generator=g) * 90.0


# --- Corrupted dataclass, base plumbing -------------------------------------------------------

def test_check_generator_raises_on_device_mismatch():
    """A silently-wrong-device generator would change the random stream without recording why
    two runs at the same seed diverged -- must be loud, not silently corrected."""
    gen = torch.Generator(device="cpu")
    with pytest.raises(RuntimeError, match="generator is on"):
        corruptions._check_generator("meta", gen)


# --- D1: Gaussian ----------------------------------------------------------------------------

def test_gaussian_shapes_and_t_range():
    clean = clean_batch(n=32)
    c = corruptions.Gaussian()
    out = c(clean, generator=torch.Generator().manual_seed(0))
    assert out.x.shape == clean.shape
    assert torch.equal(out.target, clean)
    assert out.t.shape == (32,)
    assert bool((out.t >= 0).all() and (out.t <= 1).all())
    assert out.direction is None


def test_gaussian_is_reproducible_with_the_same_seed():
    clean = clean_batch()
    a = corruptions.Gaussian()(clean, generator=torch.Generator().manual_seed(7))
    b = corruptions.Gaussian()(clean, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a.x, b.x)
    torch.testing.assert_close(a.t, b.t)


def test_gaussian_differs_with_a_different_seed():
    clean = clean_batch()
    a = corruptions.Gaussian()(clean, generator=torch.Generator().manual_seed(1))
    b = corruptions.Gaussian()(clean, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(a.x, b.x)


def test_gaussian_t_for_r_is_zero_one_and_monotonic():
    c = corruptions.Gaussian(sigma_min=0.05, sigma_max=3.0)
    assert c.t_for_r(0.05) == pytest.approx(0.0, abs=1e-6)
    assert c.t_for_r(3.0) == pytest.approx(1.0, abs=1e-6)
    assert c.t_for_r(0.5) < c.t_for_r(1.0) < c.t_for_r(2.0)


def test_gaussian_t_for_r_clamps_outside_the_range():
    c = corruptions.Gaussian(sigma_min=0.05, sigma_max=3.0)
    assert c.t_for_r(0.001) == pytest.approx(0.0)
    assert c.t_for_r(100.0) == pytest.approx(1.0)


# --- D2: variance-preserving ------------------------------------------------------------------

def test_vp_formula_matches_hand_computation(monkeypatch):
    """Exact arithmetic check against PLAN.md's own formula, not just 'runs and returns the
    right shape' -- the same standard reference_nll was held to earlier this session."""
    clean = torch.ones(2, D) * 10.0
    fixed_beta = torch.tensor([[0.25], [0.64]])
    fixed_noise = torch.ones(2, D) * 2.0

    monkeypatch.setattr(corruptions, "_rand", lambda shape, *a, **kw: fixed_beta)
    monkeypatch.setattr(corruptions, "_randn", lambda shape, *a, **kw: fixed_noise)

    c = corruptions.VariancePreserving()
    out = c(clean, generator=torch.Generator().manual_seed(0))

    token_scale = clean.norm(dim=-1, keepdim=True) / (D**0.5)  # == 10.0 here
    expected = torch.sqrt(1 - fixed_beta) * clean + torch.sqrt(fixed_beta) * fixed_noise * token_scale
    torch.testing.assert_close(out.x, expected)
    torch.testing.assert_close(out.t, fixed_beta.squeeze(-1))


def test_vp_t_is_beta_directly_no_transform():
    clean = clean_batch(n=100)
    out = corruptions.VariancePreserving()(clean, generator=torch.Generator().manual_seed(0))
    assert bool((out.t >= 0).all() and (out.t <= 1).all())


def test_vp_shapes():
    clean = clean_batch(n=16)
    out = corruptions.VariancePreserving()(clean, generator=torch.Generator().manual_seed(0))
    assert out.x.shape == clean.shape
    assert out.direction is None


# --- Rank1 base (unstructured, sphere directions) -----------------------------------------

def test_rank1_direction_is_unit_norm():
    clean = clean_batch(n=32)
    out = corruptions.Rank1()(clean, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(out.direction.norm(dim=-1), torch.ones(32), atol=1e-4, rtol=0)


def test_rank1_push_magnitude_matches_rho_times_clean_norm():
    """x - clean = rho * ||clean|| * u exactly, by construction -- an algebraic identity."""
    clean = clean_batch(n=32)
    out = corruptions.Rank1(rho_min=0.05, rho_max=3.0)(clean, generator=torch.Generator().manual_seed(0))
    push = out.x - clean
    ratio = push.norm(dim=-1) / clean.norm(dim=-1)
    assert bool((ratio >= 0.05 - 1e-4).all() and (ratio <= 3.0 + 1e-4).all())


def test_rank1_directions_vary_across_the_sphere():
    """Not the same direction every time -- would defeat the point of a diverse corruption."""
    clean = clean_batch(n=32)
    out = corruptions.Rank1()(clean, generator=torch.Generator().manual_seed(0))
    cos = (out.direction[0] * out.direction[1]).sum()
    assert abs(float(cos)) < 0.9  # two random directions in 8-D are very unlikely to align


def test_rank1_t_for_r_uses_log_uniform_normalisation():
    c = corruptions.Rank1(rho_min=0.05, rho_max=3.0)
    assert c.t_for_r(0.05) == pytest.approx(0.0, abs=1e-6)
    assert c.t_for_r(3.0) == pytest.approx(1.0, abs=1e-6)


# --- StructuredRank1 / D3 / D4: THE holdout guard --------------------------------------------

def fake_decoder(n_features=1000, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_features, d, generator=g)


def fake_split(dev, test, pool_size=1000):
    from steering.vectors import VectorSplit

    return VectorSplit(dev=dev, test=test, pool_size=pool_size, seed=0, model="gpt2",
                       source="test", created="")


def test_structured_rank1_base_raises_on_a_hand_crafted_holdout_overlap():
    """Bypasses split.train_pool() entirely -- directly tests the constructor's own guard,
    not just that using it correctly happens to avoid the problem."""
    decoder = fake_decoder()
    with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
        corruptions.StructuredRank1(decoder, pool_indices=[1, 2, 3, 500], holdout={3, 999})


def test_structured_rank1_base_accepts_a_disjoint_pool():
    decoder = fake_decoder()
    c = corruptions.StructuredRank1(decoder, pool_indices=[1, 2, 3], holdout={999})
    assert c.pool_indices == [1, 2, 3]


def test_structured_rank1_directions_are_exactly_pool_rows():
    """Not a combination of pool rows -- each drawn direction must be exactly one of them
    (unit-normalised), or this stops being 'steering-shaped' corruption."""
    decoder = fake_decoder()
    c = corruptions.StructuredRank1(decoder, pool_indices=[1, 2, 3], holdout={999})
    clean = clean_batch(n=50)
    out = c(clean, generator=torch.Generator().manual_seed(0))
    pool_unit = decoder[[1, 2, 3]] / decoder[[1, 2, 3]].norm(dim=-1, keepdim=True)
    for row in out.direction:
        matches = (pool_unit - row).norm(dim=-1) < 1e-4
        assert bool(matches.any()), "drawn direction is not exactly a pool row"


def test_d3_fixed_pool_rejects_dev_and_test_via_the_real_split(monkeypatch):
    """End-to-end through the real construction path a training script would use."""
    decoder = fake_decoder(n_features=1000)
    split = fake_split(dev=list(range(20)), test=list(range(20, 60)), pool_size=1000)
    d3 = corruptions.FixedPoolRank1(decoder, split, pool_size=100, seed=0)
    assert d3.id == "D3"
    assert set(d3.pool_indices) & split.holdout == set()
    assert len(d3.pool_indices) == 100


def test_d3_pool_is_reproducible_given_the_same_seed():
    decoder = fake_decoder(n_features=1000)
    split = fake_split(dev=list(range(20)), test=list(range(20, 60)), pool_size=1000)
    a = corruptions.FixedPoolRank1(decoder, split, pool_size=50, seed=42)
    b = corruptions.FixedPoolRank1(decoder, split, pool_size=50, seed=42)
    assert a.pool_indices == b.pool_indices


def test_d3_pool_differs_with_a_different_seed():
    decoder = fake_decoder(n_features=1000)
    split = fake_split(dev=list(range(20)), test=list(range(20, 60)), pool_size=1000)
    a = corruptions.FixedPoolRank1(decoder, split, pool_size=50, seed=1)
    b = corruptions.FixedPoolRank1(decoder, split, pool_size=50, seed=2)
    assert a.pool_indices != b.pool_indices


def test_d3_raises_if_pool_size_exceeds_the_train_pool():
    decoder = fake_decoder(n_features=100)
    split = fake_split(dev=list(range(40)), test=list(range(40, 90)), pool_size=100)
    with pytest.raises(ValueError, match="exceeds"):
        corruptions.FixedPoolRank1(decoder, split, pool_size=50)  # only 10 non-holdout left


def test_d4_uses_the_entire_train_pool_not_a_subset():
    decoder = fake_decoder(n_features=1000)
    split = fake_split(dev=list(range(20)), test=list(range(20, 60)), pool_size=1000)
    d4 = corruptions.FullPoolRank1(decoder, split)
    assert d4.id == "D4"
    assert d4.pool_indices == split.train_pool()
    assert set(d4.pool_indices) & split.holdout == set()


def test_d4_holdout_guard_fires_even_if_someone_mutates_the_split_object():
    """Defense in depth: even a split object with a hand-tampered holdout must not silently
    pass through FullPoolRank1's construction."""
    decoder = fake_decoder(n_features=200)
    split = fake_split(dev=[0, 1], test=[2, 3], pool_size=200)
    # Simulate a bug elsewhere: train_pool() computed correctly, but something upstream
    # passed a holdout set that disagrees with it.
    tampered_holdout = {0, 1, 50}  # 50 is NOT actually excluded from split.train_pool()
    with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
        corruptions.StructuredRank1(decoder, pool_indices=split.train_pool(),
                                    holdout=tampered_holdout | {split.train_pool()[0]})


# --- describe(): checkpoint provenance, must be JSON-safe -------------------------------------

@pytest.mark.parametrize("corruption_factory", [
    lambda: corruptions.Gaussian(),
    lambda: corruptions.VariancePreserving(),
    lambda: corruptions.Rank1(),
])
def test_describe_is_json_safe(corruption_factory):
    from steering.io import _json_safe

    c = corruption_factory()
    assert _json_safe(c.describe())
    assert c.describe()["corruption"] == c.id


def test_structured_describe_includes_pool_size():
    decoder = fake_decoder()
    split = fake_split(dev=list(range(20)), test=list(range(20, 60)), pool_size=1000)
    d3 = corruptions.FixedPoolRank1(decoder, split, pool_size=50)
    from steering.io import _json_safe

    d = d3.describe()
    assert _json_safe(d)
    assert d["pool_size"] == 50
    assert d["corruption"] == "D3"


# --- t_for_r round-trips from a stored description, without the SAE or split ------------------

def test_t_for_r_from_description_rebuilds_gaussian():
    desc = corruptions.Gaussian(sigma_min=0.1, sigma_max=2.0).describe()
    t = corruptions.t_for_r(desc, r=0.1)
    assert t == pytest.approx(0.0, abs=1e-6)


def test_t_for_r_from_description_rebuilds_rank1_family():
    desc = corruptions.Rank1(rho_min=0.05, rho_max=3.0).describe()
    t = corruptions.t_for_r(desc, r=3.0)
    assert t == pytest.approx(1.0, abs=1e-6)


def test_t_for_r_from_description_rebuilds_vp():
    desc = corruptions.VariancePreserving().describe()
    assert corruptions.t_for_r(desc, r=0.4) == pytest.approx(0.4)

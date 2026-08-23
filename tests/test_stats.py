"""Paired bootstrap intervals, matched-quality/matched-concept comparison, and the Pareto band
(PLAN.md Step 8's "primary results": Pareto frontier, quality at matched concept, concept at
matched quality, paired bootstrap intervals).

Ported from the exploratory repo's `evaluation/significance.py` and `evaluation/pareto.py`
(Boris: "consult previous repos where needed"), trimmed to what PLAN.md actually asks for --
four comparison methods here (B1/B2/D1/D4), not the many-arm report apparatus the source
carried. One piece of hard-won design kept intact: the band a figure draws must be built by
*exactly* the procedure that produces the paired-test p-value, or the two can disagree with
each other on the same plot. An earlier version there bootstrapped the mean front of each arm
and differenced those -- 1.4x wider than the paired construction, because resampling units
changes which units define each arm's mean front *before* interpolation runs, and the two
arms' interpolation errors stop cancelling. Within-unit differencing avoids that by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steering import stats

# --- bootstrap_ci ----------------------------------------------------------------------------

def test_bootstrap_ci_mean_matches_numpy():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = stats.bootstrap_ci(values, seed=0)
    assert mean == pytest.approx(3.0)
    assert lo <= mean <= hi


def test_bootstrap_ci_excludes_nan():
    values = np.array([1.0, 2.0, np.nan, 3.0])
    mean, _lo, _hi = stats.bootstrap_ci(values, seed=0)
    assert mean == pytest.approx(2.0)


def test_bootstrap_ci_empty_is_all_nan():
    mean, lo, hi = stats.bootstrap_ci(np.array([]), seed=0)
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_all_nan_is_all_nan():
    mean, lo, hi = stats.bootstrap_ci(np.array([np.nan, np.nan]), seed=0)
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_reproducible_with_same_seed():
    values = np.random.default_rng(1).normal(size=40)
    a = stats.bootstrap_ci(values, seed=5)
    b = stats.bootstrap_ci(values, seed=5)
    assert a == b


def test_bootstrap_ci_widens_with_more_variance():
    tight = np.array([5.0] * 20) + np.random.default_rng(0).normal(scale=0.01, size=20)
    wide = np.array([5.0] * 20) + np.random.default_rng(0).normal(scale=5.0, size=20)
    _, lo_t, hi_t = stats.bootstrap_ci(tight, seed=0)
    _, lo_w, hi_w = stats.bootstrap_ci(wide, seed=0)
    assert (hi_w - lo_w) > (hi_t - lo_t)


# --- paired_test -------------------------------------------------------------------------------

def test_paired_test_mean_diff_hand_computed():
    a = np.array([5.0, 6.0, 7.0])
    b = np.array([2.0, 2.0, 2.0])
    res = stats.paired_test(a, b, seed=0)
    assert res["mean_diff"] == pytest.approx(np.mean(a - b))
    assert res["n"] == 3
    assert res["wins"] == 3


def test_paired_test_excludes_pairs_with_either_side_nan():
    a = np.array([5.0, np.nan, 7.0, 8.0])
    b = np.array([2.0, 2.0, np.nan, 3.0])
    res = stats.paired_test(a, b, seed=0)
    # only indices 0 and 3 have both sides present
    assert res["n"] == 2
    assert res["mean_diff"] == pytest.approx(((5.0 - 2.0) + (8.0 - 3.0)) / 2)


def test_paired_test_identical_arms_give_p_equal_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    res = stats.paired_test(a, a.copy(), seed=0)
    assert res["mean_diff"] == pytest.approx(0.0)
    assert res["p"] == pytest.approx(1.0)


def test_paired_test_detects_a_consistent_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=5.0, scale=0.3, size=30)
    b = a - 2.0 + rng.normal(scale=0.1, size=30)  # consistently ~2 below a
    res = stats.paired_test(a, b, seed=0)
    assert res["p"] < 0.05
    assert res["wins"] == 30


def test_paired_test_empty_after_nan_removal():
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    res = stats.paired_test(a, b, seed=0)
    assert res["n"] == 0
    assert np.isnan(res["mean_diff"])


# --- holm --------------------------------------------------------------------------------------

def test_holm_hand_computed():
    # Standard Holm example: sorted p = [0.01, 0.02, 0.03, 0.04], n=4
    # adjusted = max_so_far(min(1, (n-i)*p_i)):
    #   i=0: min(1, 4*0.01)=0.04            -> running 0.04
    #   i=1: min(1, 3*0.02)=0.06            -> running 0.06
    #   i=2: min(1, 2*0.03)=0.06            -> running 0.06 (no decrease)
    #   i=3: min(1, 1*0.04)=0.04 -> max(0.06, 0.04) = 0.06
    pvalues = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
    adjusted = stats.holm(pvalues)
    assert adjusted["a"] == pytest.approx(0.04)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["d"] == pytest.approx(0.06)


def test_holm_never_exceeds_one():
    pvalues = {"a": 0.9, "b": 0.8, "c": 0.7}
    adjusted = stats.holm(pvalues)
    assert all(v <= 1.0 for v in adjusted.values())


def test_holm_monotonic_non_decreasing_in_sorted_order():
    pvalues = {"a": 0.001, "b": 0.4, "c": 0.2, "d": 0.05}
    adjusted = stats.holm(pvalues)
    ordered = sorted(pvalues, key=lambda k: pvalues[k])
    values = [adjusted[k] for k in ordered]
    assert values == sorted(values)


# --- value_at_target: front interpolation, either direction -----------------------------------

def make_front_df(n_units=6):
    """Two methods, several units (features), a clean synthetic quality/concept front each.
    `matched_band` needs at least 4 units for a meaningful bootstrap (ported threshold); the
    matched_comparison/value_at_target tests don't need that many, so they use the 3-unit
    default via UNITS_3 below instead of calling this directly."""
    rows = []
    for method, slope in [("A", 1.0), ("B", 0.6)]:
        for unit in range(1, n_units + 1):
            for x in (0.0, 0.5, 1.0):
                rows.append({"method": method, "unit": unit, "x": x, "y": slope * x + unit})
    return pd.DataFrame(rows)


def test_value_at_target_linear_interpolation():
    df = make_front_df()
    # method A, unit 1: y = x + 1, so at x=0.25 -> y=1.25
    y = stats.value_at_target(df, method="A", unit_value=1, target=0.25,
                              unit_col="unit", x="x", y="y")
    assert y == pytest.approx(1.25)


def test_value_at_target_nan_outside_range():
    df = make_front_df()
    y = stats.value_at_target(df, method="A", unit_value=1, target=5.0,
                              unit_col="unit", x="x", y="y")
    assert np.isnan(y)


def test_value_at_target_nan_for_missing_unit():
    df = make_front_df()
    y = stats.value_at_target(df, method="A", unit_value=999, target=0.5,
                              unit_col="unit", x="x", y="y")
    assert np.isnan(y)


def test_value_at_target_works_in_reverse_direction():
    """Swapping x/y answers the complementary question -- 'quality at matched concept' is the
    same function as 'concept at matched quality', just with the axes swapped by the caller."""
    df = make_front_df()
    # method A, unit 1: y = x + 1 -> x = y - 1, so at y=1.25 -> x=0.25
    x = stats.value_at_target(df, method="A", unit_value=1, target=1.25,
                              unit_col="unit", x="y", y="x")
    assert x == pytest.approx(0.25)


# --- matched_comparison -------------------------------------------------------------------------

def test_matched_comparison_hand_checked():
    df = make_front_df(n_units=3)
    result = stats.matched_comparison(df, "A", "B", unit_col="unit", x="x", y="y",
                                      targets=(0.5,), seed=0)
    assert len(result) == 1
    row = result.iloc[0]
    # at x=0.5: A gives y=1.5,2.5,3.5 (mean 2.5); B gives y=1.3,2.3,3.3 (mean 2.3)
    assert row["A_mean"] == pytest.approx(2.5)
    assert row["B_mean"] == pytest.approx(2.3)
    assert row["mean_diff"] == pytest.approx(0.2, abs=1e-6)
    assert row["n_paired"] == 3


def test_matched_comparison_multiple_targets():
    df = make_front_df()
    result = stats.matched_comparison(df, "A", "B", unit_col="unit", x="x", y="y",
                                      targets=(0.0, 0.5, 1.0), seed=0)
    assert len(result) == 3
    assert list(result["target"]) == [0.0, 0.5, 1.0]


# --- matched_band: the Pareto-frontier-with-CI construction ------------------------------------

def test_matched_band_too_few_units_returns_empty():
    df = make_front_df()
    small = df[df.unit == 1]
    result = stats.matched_band(small[small.method == "A"], small[small.method == "B"],
                                unit_col="unit", x="x", y="y", n_boot=100, n_grid=5, seed=0)
    assert result.empty


def test_matched_band_grid_within_overlap_of_both_fronts():
    df = make_front_df()
    arm, baseline = df[df.method == "A"], df[df.method == "B"]
    result = stats.matched_band(arm, baseline, unit_col="unit", x="x", y="y",
                                n_boot=200, n_grid=10, seed=0)
    assert not result.empty
    assert result["x"].min() >= 0.0 - 1e-9
    assert result["x"].max() <= 1.0 + 1e-9


def test_matched_band_diff_matches_the_point_estimate_at_a_shared_target():
    """The band's diff column at a grid point must agree with matched_comparison's own point
    estimate there -- the whole reason for building the band the same way the point test is
    built, not by an easier but wider separate construction."""
    df = make_front_df()
    arm, baseline = df[df.method == "A"], df[df.method == "B"]
    point = stats.matched_comparison(df, "A", "B", unit_col="unit", x="x", y="y",
                                     targets=(0.5,), seed=0).iloc[0]

    band = stats.matched_band(arm, baseline, unit_col="unit", x="x", y="y",
                              n_boot=200, n_grid=41, seed=0)  # 41 points -> 0.5 lands exactly
    nearest = band.iloc[(band["x"] - 0.5).abs().argmin()]
    assert nearest["diff"] == pytest.approx(point["mean_diff"], abs=1e-6)


def test_matched_band_n_counts_paired_units():
    df = make_front_df()
    arm, baseline = df[df.method == "A"], df[df.method == "B"]
    result = stats.matched_band(arm, baseline, unit_col="unit", x="x", y="y",
                                n_boot=100, n_grid=5, seed=0)
    assert (result["n"] == 6).all()  # all 6 units span the full [0,1] range in both arms


def test_matched_band_drops_units_missing_from_one_arm():
    df = make_front_df()
    arm = df[df.method == "A"]
    baseline = df[(df.method == "B") & (df.unit != 3)]  # unit 3 missing from baseline
    result = stats.matched_band(arm, baseline, unit_col="unit", x="x", y="y",
                                n_boot=100, n_grid=5, seed=0)
    assert (result["n"] == 5).all()  # 6 units total, 1 missing from baseline -> 5 paired

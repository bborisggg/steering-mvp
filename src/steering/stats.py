"""Paired bootstrap intervals, matched-quality/matched-concept comparison, and the Pareto band
(PLAN.md Step 8's "primary results").

**Comparisons are paired, not independent samples.** Every method here is evaluated on the same
TEST concepts at the same `r` values, so the natural unit is the within-concept difference.
Treating the two arms as independent throws that structure away and can make a real, consistent
difference look like overlapping noise.

**Bootstrap, not a normal interval.** Bounded scores (a judge's 0-100, a fire rate in [0,1])
pile up against their limits; every percentile-bootstrap resample mean is a mean of values that
were actually observed, so it can't put the interval outside the possible range the way a normal
approximation can.

**The band a figure draws must be the same construction as the point test the text quotes.**
:func:`matched_band` builds its interval from the same within-unit differencing
:func:`matched_comparison` uses at a single target, evaluated on a grid instead -- not a
separately-bootstrapped mean front for each arm, differenced afterward. An earlier version did
exactly that and it was measurably wider (1.4x) for a specific, findable reason: resampling
units changes which units define each arm's *mean* front before interpolation ever runs, so the
two arms' interpolation error stops cancelling the way it does when the difference is taken
within each unit first. See :func:`matched_band`'s own docstring for the mechanism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 5000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and a percentile bootstrap interval, NaN-safe."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def paired_test(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired difference ``a - b`` with a bootstrap interval and a sign-flip permutation p.

    The permutation flips the sign of each pair's difference independently, the exact null for
    "the two arms are exchangeable within a pair" -- no distributional assumption needed.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if d.size == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan"), "wins": 0}

    mean, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, d.size))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(d.mean())).mean())
    return {"n": int(d.size), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "p": p,
            "wins": int((d > 0).sum())}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, keyed as given. Controls the family-wise error rate
    across a table of comparisons without assuming they're independent."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (n - i) * p))
        adjusted[key] = running
    return adjusted


def value_at_target(
    df: pd.DataFrame, method: str, unit_value, target: float,
    unit_col: str, x: str, y: str,
) -> float:
    """``y`` at ``x == target``, for one unit under one method, by linear interpolation.

    NaN when ``target`` lies outside that unit's achieved range for ``x`` -- a method whose
    front never reaches that point has no value there, and extrapolating would credit it for an
    operating point it cannot actually occupy. Swap ``x``/``y`` to answer the complementary
    question: PLAN.md wants both "quality at matched concept" and "concept at matched quality",
    which are this same function called with the two columns in opposite roles.
    """
    g = df[(df["method"] == method) & (df[unit_col] == unit_value)]
    if g.empty:
        return float("nan")
    g = g.sort_values(x)
    xs, ys = g[x].to_numpy(dtype=float), g[y].to_numpy(dtype=float)
    keep = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0 or target < xs.min() or target > xs.max():
        return float("nan")
    return float(np.interp(target, xs, ys))


def matched_comparison(
    df: pd.DataFrame, a: str, b: str, unit_col: str, x: str, y: str,
    targets: tuple[float, ...], seed: int = 0,
) -> pd.DataFrame:
    """Paired difference ``a - b`` in ``y`` at each ``x`` target, over units."""
    units = sorted(df[unit_col].unique())
    rows = []
    for target in targets:
        va = np.array([value_at_target(df, a, u, target, unit_col, x, y) for u in units])
        vb = np.array([value_at_target(df, b, u, target, unit_col, x, y) for u in units])
        res = paired_test(va, vb, seed=seed)
        rows.append({
            "target": target,
            f"{a}_mean": float(np.nanmean(va)) if not np.all(np.isnan(va)) else float("nan"),
            f"{b}_mean": float(np.nanmean(vb)) if not np.all(np.isnan(vb)) else float("nan"),
            "mean_diff": res["mean_diff"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
            "p": res["p"], "wins": res["wins"], "n_paired": res["n"],
        })
    return pd.DataFrame(rows)


def matched_band(
    arm: pd.DataFrame, baseline: pd.DataFrame, unit_col: str, x: str, y: str,
    n_boot: int = 4000, n_grid: int = 24, seed: int = 0,
) -> pd.DataFrame:
    """``arm``'s Pareto front against ``baseline``, with a band built the same way the paired
    point test is: each unit's own front interpolated for both arms, the difference taken
    *within* the unit, and units resampled -- :func:`matched_comparison` evaluated on a grid
    instead of at a few named targets, so a figure and the numbers it's read against cannot
    disagree with each other.

    A prior version bootstrapped the *mean* front of each arm separately and differenced those.
    That construction is 1.4x wider for a specific, mechanical reason: resampling which units
    are included shifts each arm's mean front *before* interpolation runs, and the interpolation
    error that introduces does not cancel between two independently-resampled means the way it
    cancels when the difference is taken within each unit first, before any resampling happens.

    No multiplicity correction is applied to the band itself -- it is a pointwise interval, the
    same quantity :func:`paired_test` reports at one target. Apply :func:`holm` where a family
    of point comparisons is tabulated, not to a continuous band.
    """
    units = np.array(sorted(set(arm[unit_col]) & set(baseline[unit_col])))
    if units.size < 4:
        return pd.DataFrame()

    def unit_front(df: pd.DataFrame, u) -> tuple[np.ndarray, np.ndarray]:
        g = df[df[unit_col] == u].sort_values(x)
        xs, ys = g[x].to_numpy(dtype=float), g[y].to_numpy(dtype=float)
        keep = ~(np.isnan(xs) | np.isnan(ys))
        return xs[keep], ys[keep]

    def mean_front(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        block = df.groupby(x)[y].mean().reset_index().sort_values(x)
        return block[x].to_numpy(dtype=float), block[y].to_numpy(dtype=float)

    ax_, _ = mean_front(arm)
    bx, _ = mean_front(baseline)
    if ax_.size == 0 or bx.size == 0:
        return pd.DataFrame()
    lo, hi = max(ax_.min(), bx.min()), min(ax_.max(), bx.max())
    if not np.isfinite(lo) or hi <= lo:
        return pd.DataFrame()
    grid = np.linspace(lo, hi, n_grid)

    def curves(df: pd.DataFrame) -> np.ndarray:
        out = np.full((units.size, grid.size), np.nan)
        for i, u in enumerate(units):
            xs, ys = unit_front(df, u)
            if xs.size < 2:
                continue
            inside = (grid >= xs.min()) & (grid <= xs.max())
            out[i, inside] = np.interp(grid[inside], xs, ys)
        return out

    arm_curves, base_curves = curves(arm), curves(baseline)
    diffs = arm_curves - base_curves
    paired = ~np.isnan(diffs)
    counts = paired.sum(axis=0)
    safe = np.maximum(counts, 1)

    def mean_where_paired(a: np.ndarray) -> np.ndarray:
        masked = np.where(paired, a, np.nan)
        return np.where(counts > 0, np.nansum(masked, axis=0) / safe, np.nan)

    centre = mean_where_paired(arm_curves)
    reference = mean_where_paired(base_curves)
    diff = mean_where_paired(diffs)

    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, grid.size), np.nan)
    with np.errstate(invalid="ignore"):
        for i, rows in enumerate(rng.integers(0, units.size, size=(n_boot, units.size))):
            block = diffs[rows]
            present = (~np.isnan(block)).sum(axis=0)
            totals = np.nansum(block, axis=0)
            draws[i] = np.where(present > 0, totals / np.maximum(present, 1), np.nan)

    usable = counts > 0
    ci_lo = np.full(grid.size, np.nan)
    ci_hi = np.full(grid.size, np.nan)
    p_value = np.full(grid.size, np.nan)
    if usable.any():
        ci_lo[usable], ci_hi[usable] = np.nanpercentile(draws[:, usable], [2.5, 97.5], axis=0)
        p_value[usable] = 2 * np.minimum(
            np.nanmean(draws[:, usable] <= 0, axis=0), np.nanmean(draws[:, usable] >= 0, axis=0)
        )

    table = pd.DataFrame({
        "x": grid, "arm": centre, "baseline": reference, "diff": diff,
        "lo": centre - (diff - ci_lo), "hi": centre + (ci_hi - diff),
        "p": np.clip(p_value, 0.0, 1.0), "n": counts,
    })
    return table[table.n > 0].reset_index(drop=True)

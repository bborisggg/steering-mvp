"""Caching must be keyed by config, not by filename.

The failure this guards against is silent: a config changes, the old file is still on disk under
the same name, and it loads without complaint. The result has the right shape and plausible
values, so nothing downstream notices.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from steering import io


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point RESULTS and ARTIFACTS at a temp dir so tests never touch the real ones."""
    monkeypatch.setattr(io, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(io, "ARTIFACTS", tmp_path / "artifacts")
    return tmp_path


# --- hashing -------------------------------------------------------------------------------

def test_hash_ignores_key_order():
    assert io.config_hash({"a": 1, "b": 2}) == io.config_hash({"b": 2, "a": 1})


def test_hash_ignores_nested_key_order():
    assert io.config_hash({"x": {"a": 1, "b": 2}}) == io.config_hash({"x": {"b": 2, "a": 1}})


@pytest.mark.parametrize(
    "changed",
    [
        {"seed": 1},                       # a seed change is a different run
        {"split_hash": "deadbeef"},        # a different split is a different experiment
        {"nested": {"layer": 7}},          # nested changes must not be missed
        {"version": 2},                    # the caller's escape hatch for changed code
    ],
)
def test_hash_changes_when_any_value_changes(changed):
    base = {"seed": 0, "split_hash": "abc", "nested": {"layer": 6}, "version": 1}
    assert io.config_hash(base) != io.config_hash({**base, **changed})


def test_hash_distinguishes_int_from_string():
    """'6' and 6 must not collide -- a YAML quoting slip would otherwise be invisible."""
    assert io.config_hash({"layer": 6}) != io.config_hash({"layer": "6"})


def test_unserialisable_config_raises_and_names_the_key():
    import torch

    with pytest.raises(TypeError, match=r"config\.model\.weights"):
        io.config_hash({"model": {"weights": torch.zeros(3)}})


def test_path_in_config_raises():
    """Path reprs differ across machines, so a Path would make the hash non-portable."""
    from pathlib import Path

    with pytest.raises(TypeError, match="no stable serialisation"):
        io.config_hash({"out": Path("/tmp/x")})


# --- caching -------------------------------------------------------------------------------

def test_computes_once_then_loads(sandbox):
    calls = []

    def fn():
        calls.append(1)
        return {"value": 42}

    first = io.run_or_load("demo", {"seed": 0}, fn, quiet=True)
    second = io.run_or_load("demo", {"seed": 0}, fn, quiet=True)
    assert first == second == {"value": 42}
    assert len(calls) == 1, "second call recomputed instead of loading the cache"


def test_changed_config_raises_rather_than_silently_reusing(sandbox):
    io.run_or_load("demo", {"seed": 0}, lambda: {"v": 1}, quiet=True)
    with pytest.raises(io.CacheMismatch) as excinfo:
        io.run_or_load("demo", {"seed": 1}, lambda: {"v": 2}, quiet=True)
    assert "seed: 0 -> 1" in str(excinfo.value), "the message must name the key that changed"


def test_force_recomputes_and_rewrites_meta(sandbox):
    io.run_or_load("demo", {"seed": 0}, lambda: {"v": 1}, quiet=True)
    out = io.run_or_load("demo", {"seed": 1}, lambda: {"v": 2}, force=True, quiet=True)
    assert out == {"v": 2}
    meta = json.loads((sandbox / "results" / "demo.meta.json").read_text())
    assert meta["config"] == {"seed": 1}
    assert meta["hash"] == io.config_hash({"seed": 1})
    # And the now-valid cache loads without force.
    assert io.run_or_load("demo", {"seed": 1}, lambda: {"v": 3}, quiet=True) == {"v": 2}


def test_meta_records_the_config_for_provenance(sandbox):
    config = {"model": "gpt2", "layer": 6, "seed": 0, "version": 1}
    io.run_or_load("demo", config, lambda: {"v": 1}, quiet=True)
    meta = json.loads((sandbox / "results" / "demo.meta.json").read_text())
    assert meta["config"] == config
    assert meta["artifact"] == "demo.json"


def test_artifact_without_meta_is_treated_as_a_mismatch(sandbox):
    """A file placed by hand has no provenance, so it must not be trusted as a cache hit."""
    (sandbox / "results").mkdir(parents=True)
    (sandbox / "results" / "demo.json").write_text('{"v": 1}')
    with pytest.raises(io.CacheMismatch):
        io.run_or_load("demo", {"seed": 0}, lambda: {"v": 2}, quiet=True)


# --- serialisation -------------------------------------------------------------------------

def test_dataframe_round_trips_as_csv(sandbox):
    frame = pd.DataFrame({"alpha": [0.0, 0.5], "nll": [3.5, 4.25]})
    io.run_or_load("table", {"v": 1}, lambda: frame, quiet=True)
    assert (sandbox / "results" / "table.csv").exists(), "tables must be CSV so results/ diffs"
    loaded = io.run_or_load("table", {"v": 1}, lambda: frame, quiet=True)
    pd.testing.assert_frame_equal(loaded, frame)


def test_tensor_round_trips_and_goes_to_artifacts(sandbox):
    import torch

    tensor = torch.randn(4, 8)
    io.run_or_load("acts", {"v": 1}, lambda: tensor, where="artifacts", quiet=True)
    assert (sandbox / "artifacts" / "acts.pt").exists()
    loaded = io.run_or_load("acts", {"v": 1}, lambda: tensor, where="artifacts", quiet=True)
    assert torch.equal(loaded, tensor)


def test_results_and_artifacts_are_separate_namespaces(sandbox):
    """Same name in both roots must not collide -- results/ is committed, artifacts/ is not."""
    io.run_or_load("thing", {"v": 1}, lambda: {"where": "results"}, quiet=True)
    io.run_or_load("thing", {"v": 2}, lambda: {"where": "artifacts"},
                   where="artifacts", quiet=True)
    assert io.run_or_load("thing", {"v": 1}, lambda: None, quiet=True) == {"where": "results"}


# --- seeding -------------------------------------------------------------------------------

def test_set_seed_makes_torch_reproducible():
    import torch

    io.set_seed(0)
    a = torch.randn(16)
    io.set_seed(0)
    assert torch.equal(a, torch.randn(16))


def test_set_seed_covers_numpy_and_random():
    import random

    import numpy as np

    io.set_seed(7)
    drawn = (np.random.rand(4).tolist(), [random.random() for _ in range(4)])
    io.set_seed(7)
    assert drawn == (np.random.rand(4).tolist(), [random.random() for _ in range(4)])

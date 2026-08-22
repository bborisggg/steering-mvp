"""Config-hashed caching, seeding, and paths.

Every expensive step goes through :func:`run_or_load`. It writes two files: the artifact under a
human-readable name, and a ``.meta.json`` sidecar holding the exact config that produced it plus
that config's hash.

**The sidecar is the point.** A cache keyed only by filename returns whatever happens to be on
disk, so a config change silently reuses a stale result -- and a stale result is indistinguishable
from a correct one because it has the right shape and plausible values. Here a mismatch raises and
names the keys that changed. Recomputing then requires ``force=True``, which is a decision rather
than an accident.

The config is also the provenance record: for anything in ``results/``, the sidecar says which
model revision, hook, vector split, seed and metric version produced it.

**Function bodies are not hashed.** A closure cannot be hashed reliably, so changing a function
without changing its config will reuse the old artifact. Put a ``version`` field in the config of
anything whose implementation may change, and bump it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
ARTIFACTS = REPO_ROOT / "artifacts"
CONFIGS = REPO_ROOT / "configs"


class CacheMismatch(RuntimeError):
    """A cached artifact exists but was produced by a different config."""


def canonical(config: dict[str, Any]) -> str:
    """Deterministic JSON for hashing.

    Sorted keys so key order cannot change the hash, and no silent coercion: a config holding a
    tensor or a Path raises here rather than being stringified into something that looks stable
    and is not (``Path`` repr differs across machines; a tensor's repr is truncated).
    """
    def check(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {k: check(v, f"{path}.{k}") for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [check(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(
            f"config{path} is {type(value).__name__}, which has no stable serialisation. "
            f"Convert it first -- a Path to str, a tensor to a hash or a shape."
        )

    return json.dumps(check(config, ""), sort_keys=True, separators=(",", ":"))


def config_hash(config: dict[str, Any], length: int = 16) -> str:
    return hashlib.sha256(canonical(config).encode()).hexdigest()[:length]


def _differences(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Which keys differ, so a mismatch says what changed instead of just that it did."""
    out = []
    for key in sorted(set(old) | set(new)):
        where = f"{prefix}{key}"
        a, b = old.get(key, "<absent>"), new.get(key, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += _differences(a, b, f"{where}.")
        elif a != b:
            out.append(f"  {where}: {a!r} -> {b!r}")
    return out


def set_seed(seed: int) -> None:
    """Seed every generator this project draws from."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --- serialisation -------------------------------------------------------------------------
# Format follows from the payload type. CSV for tables because results/ is committed and must
# stay diffable; torch for anything that is weights or activations.

def _save(payload: Any, stem: Path) -> Path:
    import pandas as pd

    if isinstance(payload, pd.DataFrame):
        path = stem.with_suffix(".csv")
        payload.to_csv(path, index=False)
    elif isinstance(payload, (dict, list)):
        path = stem.with_suffix(".json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        import torch

        path = stem.with_suffix(".pt")
        torch.save(payload, path)
    return path


def _load(path: Path) -> Any:
    import pandas as pd

    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _existing(stem: Path) -> Path | None:
    for suffix in (".csv", ".json", ".pt"):
        if stem.with_suffix(suffix).exists():
            return stem.with_suffix(suffix)
    return None


def run_or_load(
    name: str,
    config: dict[str, Any],
    fn: Callable[[], Any],
    *,
    where: str = "results",
    force: bool = False,
    quiet: bool = False,
) -> Any:
    """Return the cached artifact for ``config``, or compute and cache it.

    Args:
        name: human-readable stem, e.g. ``"pareto_gpt2_c4"``. Names the file; never parse it.
        config: everything the result depends on. Include a ``version`` for anything whose
            implementation may change -- function bodies are not hashed.
        fn: zero-argument callable producing the artifact.
        where: ``"results"`` (committed, small, diffable) or ``"artifacts"`` (large, gitignored).
        force: recompute and overwrite even if a valid cache exists.

    Raises:
        CacheMismatch: an artifact exists under this name but a different config, and ``force``
            is not set. The message lists the keys that differ.
    """
    root = {"results": RESULTS, "artifacts": ARTIFACTS}[where]
    root.mkdir(parents=True, exist_ok=True)
    stem = root / name
    meta_path = stem.with_suffix(".meta.json")
    digest = config_hash(config)

    existing = _existing(stem)
    if existing and not force:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("hash") == digest:
            if not quiet:
                print(f"cached  {existing.name}  ({digest})")
            return _load(existing)
        diffs = _differences(meta.get("config", {}), config) or ["  (no recorded config)"]
        raise CacheMismatch(
            f"{existing.name} was produced by a different config.\n"
            + "\n".join(diffs)
            + "\n\nPass force=True to recompute and overwrite, or rename the run."
        )

    payload = fn()
    path = _save(payload, stem)
    meta_path.write_text(
        json.dumps({"hash": digest, "artifact": path.name, "config": config}, indent=2,
                   sort_keys=True) + "\n"
    )
    if not quiet:
        print(f"computed {path.name}  ({digest})")
    return payload

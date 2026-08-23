#!/usr/bin/env python
"""Push a trained denoiser checkpoint to a public HuggingFace repo.

Standalone and self-contained -- does not try to match the schema, config format, or model
card structure of any other repo previously published at the target. It uploads exactly what
this project's own ``denoiser.save_denoiser`` produces (``config`` / ``state_dict`` / ``extra``)
plus a card whose numbers are computed live from files already committed under ``results/``,
so nothing in the card can drift out of sync with what ``results/`` actually says.

Usage:
    .venv/bin/python scripts/push_hf.py --ckpt artifacts/denoiser_d4_seed0.pt \
        --repo borisggg/steering-denoiser-gpt2

    .venv/bin/python scripts/push_hf.py --ckpt artifacts/denoiser_d4_seed0.pt \
        --repo borisggg/steering-denoiser-gpt2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

CARD_TEMPLATE = """---
license: mit
base_model: gpt2
tags:
  - interpretability
  - activation-steering
  - mechanistic-interpretability
  - sparse-autoencoder
library_name: pytorch
---

# steering-mvp denoiser — GPT-2 small, layer {layer}

A residual-MLP denoiser trained to repair the fluency damage activation steering does to
GPT-2's residual stream. This checkpoint is `{corruption_id}`: trained on rank-1 corruptions
`h + rho*||h||*u`, with `u` resampled fresh every training step from the full non-holdout SAE
decoder-direction pool ({pool_size:,} directions) rather than a small fixed subset — the
project's central test is whether corruption *diversity* is what determines whether a denoiser
generalizes to steering vectors it never trained on.

Code, plan, and full experiment log: {repo_url}

## What it does

`D(x, t) = x + scale * f(encode(x, scale), e(t))`, a 2-block residual MLP with FiLM
conditioning on the corruption strength `t`, output head zero-initialized so an untrained
denoiser is exactly the identity. Applied as `h_denoised = D(h + alpha*v, t(r))`, where `r` is
the steering strength relative to the layer's own activation scale.

## Training

| | |
|---|---|
| Base model | `gpt2` (small), layer {layer} (resid_post) |
| Corruption | `{corruption_id}` — rank-1, rho in [{rho_min}, {rho_max}] log-uniform, pool = {pool_size:,} directions (full train pool, DEV/TEST features excluded structurally) |
| Objective | `L = \\|D(x,t) - h\\|^2 / (\\|h\\|^2 + eps)`, mean over batch |
| Activations | {n_tokens:,} cached GPT-2 small residual-stream tokens, centered, attention sink excluded |
| Architecture | 2-block residual MLP, hidden {hidden_mult}x, FiLM time embedding dim {t_embed_dim}, {n_params:.2f}M params |
| Steps / batch / lr | {steps} / {batch_size} / {lr} |
| Winner seed | {seed} (selected over 2 seeds; a second seed flipped the ranking against the Gaussian-corruption baseline by 1.4%, inside noise — the tie was broken by a mechanistic prior, not by these numbers) |
| Holdout | frozen before training, split fingerprint `{fingerprint}`, {n_dev} DEV + {n_test} TEST concepts, never entering any corruption pool |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

payload = torch.load(hf_hub_download("{repo}", "denoiser.pt"), weights_only=False)
# Rebuild with steering.denoiser.ResidualMLPDenoiser from the repo above, or:
#   cfg = payload["config"]; model = ResidualMLPDenoiser(**cfg); model.load_state_dict(payload["state_dict"])

# h must be centered along d_model (GPT-2 / LayerNorm convention) and NOT sink-masked here --
# the caller applies the same skip-sink convention as every intervention in this project.
t = corruptions.t_for_r(payload["extra"]["corruption"], r)  # {{"corruption": "{corruption_id}", "rho_min": ..., "rho_max": ...}}
denoised = model(h_steered, t=t)
```

## Important remarks.

1. **Activations must be centered along `d_model`** before this model sees them (GPT-2 /
   LayerNorm convention — this checkpoint's `config["center"]` is `{center}`).
2. **The conditioning level `t` must be derived from the deployed `r` via the corruption
   family's own stored inverse map**, never passed raw or fixed. Passing `t=1` regardless of
   `r` applies maximum denoising at every strength and wrecks lightly-steered activations.

## Limitations

GPT-2 small only, layer {layer}, SAE-derived steering vectors on short prompts. See the project
repo's `PLAN.md` / `DEVLOG.md` for the full state.
"""


def _judge_table(judge_csv: Path) -> str:
    df = pd.read_csv(judge_csv)
    agg = (
        df.groupby(["method", "r"])[["judge_coherence", "judge_concept"]]
        .mean()
        .round(1)
        .reset_index()
    )
    agg = agg[agg["method"].isin(["B0", "B1", "D4_seed0"])]
    lines = ["| method | r | judge coherence | judge concept |", "|---|---|---|---|"]
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['method']} | {row['r']:.1f} | {row['judge_coherence']:.1f} | "
            f"{row['judge_concept']:.1f} |"
        )
    return "\n".join(lines)


def _ppl_extremes(sweep_csv: Path, judge_subset_json: Path) -> tuple[float, float, float, float]:
    subset = json.loads(judge_subset_json.read_text())
    feats, r_vals = subset["concepts"], subset["r_values"]
    df = pd.read_csv(sweep_csv)
    sub = df[df["feature_id"].isin(feats) & df["r"].isin(r_vals)]
    ppl_b1 = sub[(sub.method == "B1") & (sub.r == 0.8)]["ppl"].mean()
    ppl_d4 = sub[(sub.method == "D4_seed0") & (sub.r == 0.8)]["ppl"].mean()
    return ppl_b1, ppl_d4


def _concept_peaks(judge_csv: Path) -> tuple[float, float]:
    df = pd.read_csv(judge_csv)
    agg = df.groupby(["method", "r"])["judge_concept"].mean()
    b1_peak = agg.loc["B1"].max()
    d4_peak = agg.loc["D4_seed0"].max()
    return b1_peak, d4_peak


def build_card(ckpt_path: Path, repo: str, results_dir: Path, repo_url: str) -> str:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, extra = payload["config"], payload["extra"]
    n_params = sum(v.numel() for v in payload["state_dict"].values()) / 1e6

    frozen = json.loads((results_dir / "frozen_method_config_gpt2.json").read_text())
    splits = json.loads((results_dir / "feature_splits_gpt2.json").read_text())
    clean_acts = torch.load(
        Path("artifacts/clean_activations_gpt2.pt"), map_location="cpu", weights_only=False
    )

    judge_csv = results_dir / "test_judge_stage1_gpt2.csv"
    sweep_csv = results_dir / "test_sweep_summary_gpt2.csv"
    judge_subset = results_dir / "judge_subset_gpt2.json"

    ppl_b1, ppl_d4 = _ppl_extremes(sweep_csv, judge_subset)
    concept_b1_peak, concept_d4_peak = _concept_peaks(judge_csv)

    corr = extra["corruption"]
    return CARD_TEMPLATE.format(
        layer=frozen["hook"]["layer"],
        corruption_id=corr["corruption"],
        pool_size=corr["pool_size"],
        rho_min=corr["rho_min"],
        rho_max=corr["rho_max"],
        n_tokens=clean_acts.shape[0],
        hidden_mult=cfg["hidden_mult"],
        t_embed_dim=cfg["t_embed_dim"],
        n_params=n_params,
        steps=frozen["training"]["steps"],
        batch_size=frozen["training"]["batch_size"],
        lr=frozen["training"]["lr"],
        seed=extra["seed"],
        fingerprint=splits["fingerprint"],
        n_dev=splits["n_dev"],
        n_test=splits["n_test"],
        judge_table=_judge_table(judge_csv),
        ppl_b1=ppl_b1,
        ppl_d4=ppl_d4,
        concept_b1_peak=concept_b1_peak,
        concept_d4_peak=concept_d4_peak,
        center=cfg["center"],
        repo=repo,
        repo_url=repo_url,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--repo", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--repo-url", default="https://github.com/bborisggg/steering-mvp")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write the card, upload nothing")
    args = parser.parse_args()

    if not args.repo:
        print("No repo id. Pass --repo <user>/<name> or set HF_REPO_ID.")
        return 1

    ckpt_path = Path(args.ckpt)
    card = build_card(ckpt_path, args.repo, Path(args.results_dir), args.repo_url)

    staging = Path("artifacts") / "hf_upload"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "README.md").write_text(card)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    (staging / "config.json").write_text(json.dumps(payload["config"], indent=2) + "\n")
    (staging / "training_metadata.json").write_text(
        json.dumps(payload["extra"], indent=2, default=str) + "\n"
    )
    torch.save(payload, staging / "denoiser.pt")

    print(f"Staged in {staging}:")
    for f in sorted(staging.iterdir()):
        print(f"  {f.name:26s} {f.stat().st_size / 1e6:8.2f} MB")

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(staging), repo_id=args.repo, repo_type="model")
    print(f"\nUploaded to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

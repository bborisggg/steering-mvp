# steering-mvp

Does the *geometry* of the corruption a denoiser trains on decide whether it can repair
activation steering?

Steering an LLM by `h + alpha*v` gets the concept in and the fluency out. The usual fix is to
train a denoiser on `h + noise` and apply it after steering. This repo tests one idea: steering
is a **structured rank-1 perturbation**, so a denoiser trained on many *diverse* rank-1
corruptions should generalise to unseen steering vectors better than one trained on generic
noise, or on a small fixed set of directions.

Primary setting: GPT-2 small + OpenAI SAE. External validation: Gemma-2-2B-it + 6 persona traits.

**Report: [`REPORT.md`](REPORT.md).** GPT-2 headline, mechanism analysis, the PDS diagnostic,
and the Gemma-2-2B-it external validation against a prompting baseline.

## Structure

- **`PLAN.md`** — the experiment, step by step
- **`REPORT.md`** — the write-up
- **`notebooks/`** — every run is invoked from here (`01_baseline` → `04_analysis`); orchestration
  only, nothing important is defined inline
- **`src/steering/`** — all reusable code: hooks, corruptions, denoiser, interventions,
  generation, metrics, judge, stats, io
- **`scripts/`** — one-off utilities (HF checkpoint push, Gemma trait-data prep)
- **`tests/`** — run via `pytest`
- **`results/`** — committed, small, diffable outputs (CSV/JSON); every number in `REPORT.md`
  traces to a file here
- **`docs/figures/`** — generated plots
- **`configs/`** — prompt sets and persona-trait definitions

## Checkpoints

- [borisggg/steering-denoiser-gpt2](https://huggingface.co/borisggg/steering-denoiser-gpt2) —
  `D4` (rank-1, full-pool corruption over the SAE dictionary)
- [borisggg/steering-denoiser-gemma2_2b](https://huggingface.co/borisggg/steering-denoiser-gemma2_2b) —
  `D4` (rank-1, full-pool corruption over Gemma Scope's decoder pool)

Both pushed via `scripts/push_hf.py`; each model card states plainly what is and isn't
validated for that model.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[sae,dev]"
.venv/bin/pytest
```

MPS only — no CUDA, no bitsandbytes, no flash-attn.

## Status

GPT-2: corruption ablation, denoiser training, held-out test, mechanism analysis, and the PDS
diagnostic are complete. Gemma-2-2B-it external validation (persona-vector steering,
`B0`/`B1`/`D1`/`D4`, plus a prompting baseline) is complete — see `REPORT.md` §9 for where it
agrees and disagrees with the GPT-2 result.

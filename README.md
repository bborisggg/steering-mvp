# steering-mvp

Does the *geometry* of the corruption a denoiser trains on decide whether it can repair
activation steering?

Steering an LLM by `h + alpha*v` gets the concept in and the fluency out. The usual fix is to
train a denoiser on `h + noise` and apply it after steering. This repo tests one idea: steering
is a **structured rank-1 perturbation**, so a denoiser trained on many *diverse* rank-1
corruptions should generalise to unseen steering vectors better than one trained on generic
noise, or on a small fixed set of directions.

- **`PLAN.md`** — the experiment, step by step
- **`DECISIONS.md`** — frozen choices, each with its reason
- **`DEVLOG.md`** — running notes, measured timings, traps already paid for
- **`notebooks/`** — every run is invoked from here
- **`src/steering/`** — all reusable code; nothing important is defined in a notebook

Primary setting GPT-2 small + OpenAI SAE; external validation Gemma-2-2B-it + 6 persona traits.

**Checkpoint:** [borisggg/steering-denoiser-gpt2](https://huggingface.co/borisggg/steering-denoiser-gpt2)
— `D4` (rank-1, full-pool corruption), pushed via `scripts/push_hf.py`. 

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[sae,dev]"
```

MPS only — no CUDA, no bitsandbytes, no flash-attn.

## Status

Structure only. Nothing implemented yet.

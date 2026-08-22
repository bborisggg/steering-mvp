# steering-mvp — final plan

## Goal

Build the smallest clean experiment that tests one main idea:

> Steering is a structured rank-1 perturbation. A denoiser trained on many diverse rank-1 corruptions should generalize to unseen steering vectors better than generic-noise denoisers or small fixed direction pools.

Primary setting: **GPT-2 small + OpenAI SAE**.  
External validation: **Gemma-2-2B-it + 6 persona traits**.

No LoRA, no architecture sweep, no decay schedule. PDS is only a one-shot diagnostic.

---

## 1. Methods

Compare:

- **B0 — clean:** `h`
- **B1 — plain steering:** `h + αv`
- **B2 — norm control:** plain steering followed by residual-norm matching
- **D1 — Gaussian denoiser:** train on `h + σε`
- **D2 — variance-preserving noise:** `sqrt(1-β)h + sqrt(β)ε`
- **D3 — rank-1 / 256 directions:** train on `h + au`, `u` from a fixed 256-vector pool
- **D4 — rank-1 / large pool:** same corruption, with directions sampled from the full allowed SAE dictionary

D4 training pool must exclude all DEV and TEST vectors.

### Steering strength

Use relative strength

\[
r=\frac{\|\alpha v\|}{\mathbb E\|h\|},
\qquad
\alpha=r\frac{\mathbb E\|h\|}{\|v\|}.
\]

Initial grid:

`0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0`

Trim only clearly unusable extremes during baseline validation.

### Intervention positions

Use **all positions**: prompt + generated-token forwards.

Reason: previous experiments show that the prompt contributes materially through the KV cache.

---

## 2. Data split

Probe about **1024 SAE features** and apply one frozen usability rule.

Target roughly **150–200 usable concepts**.

Create disjoint sets:

- **TRAIN** — rank-1 corruption directions only
- **DEV ~50 concepts** — method selection
- **TEST 100+ concepts** — final evaluation only

Hard checks:

```text
TRAIN ∩ DEV = ∅
TRAIN ∩ TEST = ∅
DEV ∩ TEST = ∅
```

Save IDs and hashes in `artifacts/vector_split.json`.

No choice may depend on TEST results.

---

## 3. Denoiser

Use one small residual MLP:

```text
normalize
Linear(d → 2d)
SiLU/GELU
Linear(2d → 2d)
SiLU/GELU
Linear(2d → d), zero-init
residual add
```

Condition on corruption strength.

Loss:

\[
L=
\frac{\|D(x,s)-h\|_2^2}{\|h\|_2^2+\epsilon}.
\]

Monitor `||D(h,0)-h||`; add no extra regularizer unless clean states are measurably damaged.

For rank-1 training, sample relative corruption magnitude

\[
\rho=\frac{\|au\|}{\|h\|}
\]

log-uniformly over approximately `[0.05, 3]`.

---

## 4. Metrics

### Cheap quality axis

Primary metric: **reference-model NLL** on the generated continuation, conditioned on the original prompt and scored by the unmodified base LM.

Also record:

- distinct-1/2/3
- repetition rate
- length
- EOS rate

Distinct-n is diagnostic only.

### Concept axis

Use an independent concept judge/rubric.

Do not use the steered SAE feature activation as the headline concept metric; it is allowed only for mechanism analysis.

### Judge budget

Run all TEST concepts on the cheap axis at all `r`.

For expensive judging, pre-declare:

- a **random 40-concept subset of TEST**
- **4 fixed `r` values** covering the useful frontier

This keeps judge cost near the previous ~3.6 GPU-hour scale.

Before final evaluation, manually inspect a small sample and calibrate **reference NLL against judge coherence** so the cheap quality axis has an interpretable scale.

---

## 5. Implementation

```text
notebooks/
  01_baseline.ipynb
  02_train.ipynb
  03_results.ipynb
  04_analysis.ipynb

src/steering/
  hooks.py
  spaces.py
  vectors.py
  corruptions.py
  denoiser.py
  interventions.py
  generate.py
  metrics.py
  judge.py
  stats.py
  io.py

tests/
configs/
results/
artifacts/
REPORT.md
```

Required implementation:

- exact GPT-2/Gemma hook conventions
- explicit activation `encode/decode`
- vector loading + train/dev/test leakage checks
- Gaussian / VP / rank-1 corruptions
- residual denoiser
- B0/B1/B2 + denoised steering
- paired deterministic generation
- reference-model NLL
- cached judge
- paired bootstrap / matched-fluency analysis
- config-hashed caching

Important tests:

- correct hook/layer
- hook fires throughout autoregressive generation
- activation encode/decode round-trip
- denoiser is identity at initialization
- TEST vectors never enter training corruption pools
- B2 preserves the intended norm-control definition
- NLL is scored under the clean reference model
- cache keys include split/config hashes

---

# Step-by-step execution

## Step 1 — reproduce only the old baseline

Implement infrastructure and run **B0/B1 only**.

The gate is:

> clean generation and plain steering reproduce previous baseline behavior closely enough.

Do **not** expect old denoiser results to reproduce, because this MVP changes the strength convention, corruption parameterization, and loss.

If B0/B1 disagree materially with the old repo, fix the pipeline before training.

---

## Step 2 — create the larger concept split

Probe ~1024 SAE features.

Freeze the usability criterion, then create TRAIN / DEV / TEST and save their hashes.

Do not revisit the split later.

---

## Step 3 — validate plain steering

On DEV, run B0/B1/B2 across the `r` grid.

Require a clear concept–quality tradeoff.

Also calibrate reference NLL against judge coherence on a small sample.

---

## Step 4 — cache clean activations

Cache GPT-2 residual activations at the frozen hook.

All training/inference normalization must pass through the same `spaces.py` transform.

---

## Step 5 — train corruption candidates

Use one fixed denoiser architecture.

First pass, one seed:

- D1 Gaussian
- D2 variance-preserving
- D3 rank-1 / 256
- D4 rank-1 / large pool

Use this seed only to eliminate clearly weak families.

Then take the **top two** and train a **second seed** before choosing the winner.

Do not freeze a corruption family from one seed.

---

## Step 6 — small pool-size check

Only if rank-1 corruption is among the winners.

Evaluate:

`64, 256, 1024, full allowed pool`

on the same DEV concepts.

Four points are enough.

Goal: test whether held-out steering improves as corruption-direction diversity increases.

---

## Step 7 — freeze the method

Freeze:

- corruption family
- pool size
- architecture
- training distribution
- hook
- `r` grid
- intervention positions
- metrics
- judge subset and 4 judge `r` values

Save one config hash.

No changes based on TEST.

---

## Step 8 — final GPT-2 test

Train the selected denoiser with **3 seeds**.

Evaluate all TEST concepts on the cheap axis.

Headline comparison:

- B1 plain
- B2 norm control
- D1 Gaussian
- selected rank-1 denoiser

Judge only the pre-declared 40 TEST concepts × 4 `r` values.

Primary results:

- Pareto frontier
- quality at matched concept
- concept at matched quality
- paired bootstrap intervals

---

## Step 9 — mechanism analysis

Keep only two required analyses.

### A. Pool memorization/generalization

Compare reconstruction on:

- seen rank-1 directions
- unseen rank-1 directions

as pool size grows.

### B. What does the denoiser repair?

Measure:

- clean / steered / denoised PCA-distance to the clean activation distribution
- signed correction along the steering direction, `Δ · v̂`

Do not assume the sign is universal: previous GPT-2 and Gemma results differ.

---

## Step 10 — PDS diagnostic

Run the already implemented projected-denoising equivalent once.

Test:

> Is PDS explained by moving to another effective `r` on the existing steering/denoising frontier?

If yes, drop it except for a short negative-result note.

No λ sweep and no attempt to rescue it.

---

## Step 11 — Gemma external validation

Use the frozen GPT-2 recipe.

Evaluate exactly **6 persona traits**.

Compare:

- plain steering
- norm control
- Gaussian denoising
- selected rank-1 denoising

Do not redo corruption or architecture search on Gemma.

This is validation, not a second development loop.

---

# Deliverables

## Main figures

1. **GPT-2 held-out Pareto frontier:** plain, norm control, Gaussian, best rank-1
2. **Pool-size generalization:** held-out improvement vs number of corruption directions
3. **Mechanism:** seen/unseen reconstruction + activation-distribution repair
4. **Gemma:** compact frozen-recipe validation on 6 traits

## Report claim

If supported:

> Diverse rank-1 corruption is a better training distribution for repairing large steering interventions than generic noise or a small fixed direction pool. The gain generalizes to unseen steering vectors, improves the held-out concept–quality frontier beyond norm control, and transfers at least partially to Gemma persona steering.

---

# Time / compute cap

Keep this as a short MVP.

Target execution:

- infrastructure + baseline: **a few hours**
- GPT-2 denoiser selection: **<1 hour of training**
- GPT-2 cheap-axis evaluation + analysis: **a few hours**
- judge subset: **~3.6 GPU-hours**
- Gemma: one targeted frozen-recipe validation only

Aim for **one focused build/evaluation cycle plus one overnight run**, not a multi-day sweep.

If time is tight, cut in this order:

1. D2 if clearly weak after first seed
2. 64/1024 intermediate pool sizes
3. PDS diagnostic
4. extra Gemma generations

Never cut:

- B2 norm control
- train/dev/test split
- two-seed confirmation before freezing
- Gaussian baseline
- large-pool rank-1 denoiser
- held-out GPT-2 TEST
- pre-declared judge subset

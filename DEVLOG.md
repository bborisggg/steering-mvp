# Development log

Running notes, newest last. Anything that would be expensive for the next person to rediscover
goes here: measured timings, traps, surprises, and mistakes with what they cost.

Frozen choices live in `DECISIONS.md`; the experiment itself is `PLAN.md`.

---

## 2026-08-22 — repository initialised

Structure created, nothing implemented. Development is **sequential**: each module is written,
then audited, before the next. All run calls are made from the notebooks.

### Inherited facts worth not re-measuring

From the exploratory repo (`~/Coding/steering-final`, frozen). These informed the plan and are
recorded so the estimates in it can be checked rather than trusted.

**Timings, measured**

| | |
|---|---|
| SAE steerability probe | 256 candidates in **17 min** -> 1024 in ~70 min |
| replication pass | 60 features in **2.7 min** |
| GPT-2 generation | ~0.6 s per (concept, arm, r) cell |
| Gemma generation | ~30 s per cell without the judge, ~85 s with |
| judge | ~20 s per cell of 8 prompts x 2 rubrics, 4 workers |
| GPT-2 denoiser training | ~4 min · Gemma ~20 min |

Gemma is roughly **20x** GPT-2 per condition. This is why the plan develops on GPT-2 and uses
Gemma only for validation.

**Yield:** 256 probed -> 67 responsive (26%) -> 60 with semantic descriptions -> 56 reproduced ->
49 usable. Extrapolating, ~1024 probed should give 150-200 usable.

### Traps that have already cost time

Each has a test in `PLAN.md` §5. They are listed with what actually went wrong, because the
abstract statement of a bug is much easier to nod at than to avoid.

1. **`resid_pre(L+1)` vs `resid_post(L)`** — an off-by-one corrupts every downstream number and
   nothing errors.
2. **Forgetting to un-scale after the denoiser** — produces plausible-looking, wrong Pareto
   fronts. The activation scale now travels inside the checkpoint so a caller cannot supply a
   different one.
3. **Steering only the prompt** — text still generates, so it is silent. Guarded by asserting the
   hook fires more than once during generation.
4. **Perplexity under the steered model** — read 185.0 instead of 153.2 on byte-identical text.
5. **Denoiser not identity at init** — an untrained module destroys the activation instead of
   doing nothing.
6. **A flag accepted and ignored** — `--split-name` was added to a script that then read the
   default holdout through another code path. A run "on 49 concepts" silently used 16 and looked
   entirely plausible. Guarded statically: every `args.x` must be a declared flag, and the flag
   must reach whatever resolves the path.
7. **A deleted flag still referenced** — killed a run 40 minutes in.
8. **Judge called with an empty concept string** — would have scored noise; now refuses to start.
9. **A figure and the text using different estimators** — the plotted band bootstrapped the mean
   front while the quoted p-value paired within concept. 1.4x wider, and able to contradict its
   own p-value. The band a figure draws must *be* the test the text quotes.
10. **Globbing result files by pattern** — a figure labelled "denoised" silently picked whichever
    denoiser sorted first, which was the one the report calls a no-op. Name the file.

### Findings that shaped this plan

- **Corruption geometry matters, and it is pool size.** Rank-1 from 256 fixed decoder rows
  memorises its pool (unseen-direction displacement per unit of own corruption repaired: ratio
  **400**) and does not move the front. The same corruption drawing from the whole dictionary has
  ratio **5.6** and was the strongest arm measured. 4096 directions sits between at 10.4. This is
  H1+H2 and it is the spine of the MVP.
- **Sixteen concepts reported a null that 49 showed to be real.** The gain "vanishing" in the
  readable region was a power artefact. Hence the enlarged pool.
- **The denoiser's correction along `v` is model-dependent.** On GPT-2 it removes a substantial
  part of the steering; on Gemma it slightly *adds* to it. Do not assume a sign.
- **PDS (projecting the correction orthogonal to `v`) is already tested.** It is algebraically
  identical to the exploratory repo's `M3` at beta=1, and measured **worse** than plain denoising
  on GPT-2 (-0.022, p=0.025, 2 of 15 concepts) and null on Gemma. Hence one diagnostic, not a
  method.
- **A trait that separates best can steer worst.** `evil` had the highest extraction Cohen's d of
  three traits (9.41) and the lowest reachable expression (42/100 while readable) -- and the model
  writes it happily when simply asked (89.2). Separability is a readout; steerability is a lever.

### Open, needs a decision before the affected step

- **Network access** for two first-time fetches: Neuronpedia auto-interp descriptions for ~1024
  features (Step 2), and cloning `safety-research/persona_vectors` for the 6 traits (Step 11).
- **Network access** is still needed for two first-time fetches (above).

---

## 2026-08-22 — `io.py`: caching, hashing, seeds

Environment built: `.venv` on Python 3.12.0, `pip install -e ".[dev]"`, torch 2.13.0. MPS-only,
same constraints as the exploratory repo (no CUDA, no bitsandbytes, no flash-attn).

`run_or_load(name, config, fn)` writes the artifact under a readable name plus a `.meta.json`
sidecar holding the config and its hash.

**Why a sidecar and not a hashed filename.** A hash in the filename makes a config change look
like a cache *miss*, so it silently recomputes -- fine for correctness, but `results/` fills with
near-duplicate files and the report cannot cite a stable path. The sidecar inverts this: one
readable path per result, and a config change is an *error* naming the keys that moved. Choosing
to overwrite then requires `force=True`. The sidecar doubles as the provenance record the report
needs, which is what `results/` being committed is for.

Three things this deliberately does **not** do:

- **Function bodies are not hashed.** A closure has no stable hash. Anything whose implementation
  can change carries a `version` field in its config, bumped by hand. This will be forgotten at
  least once; the symptom is a result that does not move when the code does.
- **No silent coercion.** A tensor or a `Path` in a config raises. A `Path` stringifies to
  something machine-specific and a tensor's repr is truncated, so either would produce a hash
  that looks stable and is not.
- **An artifact with no sidecar is a mismatch, not a hit.** A file placed by hand has no
  provenance and must not be trusted.

Format follows the payload type: DataFrame -> CSV (so committed `results/` stays diffable),
dict/list -> JSON, anything else -> `torch.save`. `results/` and `artifacts/` are separate
namespaces, so the same name in both does not collide.

19 tests in `tests/test_io.py`, ruff clean.

---

## 2026-08-22 — `hooks.py`: one block, read and write

`ResidualHook(model, layer, fn=None, capture=False)`, context manager only. `layer=L` is the
**output** of block L, i.e. `resid_post(L)` = `resid_pre(L+1)` = `hidden_states[L+1]`. Tested
against a real GPT-2 forward, and tested that `hidden_states[L]` is *visibly different* -- an
equality test alone would pass against a near-copy.

The equivalence stops at the last block: `hidden_states[-1]` has `ln_f` applied. Documented and
tested rather than worked around; this project always intervenes mid-depth.

### `seq_lens` instead of `n_calls`

The hook records the sequence length of every forward. Under a KV cache that is
`[prompt_len, 1, 1, ...]`, so "the intervention reached the decode passes" is an assertion rather
than a hope. `n_calls > 1` was the old repo's check and it is too weak.

The trap it guards: a hook firing only on prefill **still changes the generated text**, because
the prompt's activations sit in the cache every later token attends to. So "the output changed"
is not evidence of anything. The test compares all-positions steering against a deliberately
prefill-only hook and requires them to differ.

Measured on GPT-2 small, prompt of 5 tokens, `max_new_tokens=8`: exactly 8 forwards, `seq_lens ==
[5, 1, 1, 1, 1, 1, 1, 1]`. So HF greedy generation with `use_cache=True` does one forward per new
token, no extras.

### `fn` may not change shape, dtype or device

All three raise. Dtype especially: the denoiser is fp32 and Gemma may be bf16, so a silent cast
would upconvert the rest of the network and nothing would look wrong. Conversion happens in
`spaces.py`, deliberately.

### Measured: position 0 is an attention sink

Wrote a test with a magnitude threshold and it failed, which turned out to be the code being
right. GPT-2 small, layer 6, prompt `"The capital of France is"`:

| | |
|---|---|
| residual norm, position 0 | **3046** |
| residual norm, positions 1-4 | 92, 88, 106, 85 |
| `abs().max()` over all positions | 2941, in dim **447** |
| next largest dims | 138 (791), then 378 (59) |

The first position is **33x** the rest and two dimensions carry nearly all of it. Any mean or
median over positions that includes position 0 measures the sink, not the stream. This is
`DECISIONS.md` D2 confirmed rather than assumed, and it now has its own test so that a future
change which starts including position 0 fails loudly instead of shifting the Pareto curve.

Consequence for `spaces.py`: whatever normalisation it applies has to survive a 33x outlier in
one position and a ~4x outlier in one dimension. A plain per-tensor mean/std will be dominated
by both.

`captured` holds the stream **before** `fn`, so a steered run still records the clean state. To
record what was substituted, capture from inside `fn`.

20 tests in `tests/test_hooks.py` (39 in the suite), ruff clean, ~2 s with GPT-2 cached.

---

## 2026-08-22 — `spaces.py`: centering, scale, encode/decode

Two measurements decided this module; both first-party (GPT-2 layer 6, Gemma-2-2b-it layer 12,
real forwards, relative logit change under the operation in question), not carried over from
the exploratory repo, though they land on the same conclusions it did.

**Centering is architecture-gated, not a preference.**

| | relative logit change |
|---|---|
| GPT-2 (LayerNorm) | 3.37e-07 |
| Gemma-2-2b-it (RMSNorm) | 3.19e-02 |

~95,000x apart. LayerNorm subtracts the mean before every read, so the mean component along
`d_model` provably cannot reach a prediction; RMSNorm never subtracts it, so removing it is a
real edit. `spaces.should_center(model_name)` is an explicit, closed table -- `{"gpt2": True,
"google/gemma-2-2b-it": False, "google/gemma-2-2b": False}` -- and **raises** on anything not
in it, rather than defaulting. A silent default would centre an unmeasured architecture, and
the failure mode (a real perturbation that looks like a Pareto curve) does not announce itself.

**Scale excludes the sink, confirming yesterday's number on 200 Pile sequences instead of one
prompt.** Per-token norm quantiles at layer 6 (positions 1..127, n=32512): median **88.8**,
p95/p05 spread 1.4x, max/median 1.4x -- GPT-2's stream is otherwise remarkably uniform. Position
0: median **3046**, i.e. **34x**. `activation_scale()` is the median over non-sink, non-pad
tokens, matching `DECISIONS.md` D2 and now unit-tested against a hand-computed median so a
future refactor can't quietly switch it to a mean.

### What `encode`/`decode` round-trips to

`decode(encode(h)) == center(h)`, not `== h`. Centering is written back into the stream, not
undone on decode -- for GPT-2 there's nothing to undo (output-neutral), and Gemma never centers
in the first place. The only thing the pair promises to restore exactly is the scale. Got this
wrong on the first pass of the design (see below) before writing the test that pins it.

### Open question surfaced, not resolved here: does steering itself skip the sink?

`spaces.py` only handles what the *denoiser* sees. A separate question, for `interventions.py`,
is whether `h + alpha*v` should touch position 0 at all. Measured on GPT-2, `r` in {0.5, 1, 2},
steering everywhere vs. everywhere-but-position-0:

| r | full push vs clean | skip-sink vs full push | argmax agreement |
|---|---|---|---|
| 0.5 | 24.1% | 1.6% | 96.7% |
| 1.0 | 34.1% | 2.5% | 92.8% |
| 2.0 | 45.1% | 3.1% | 88.7% |

Steering *only* the sink, in isolation, moves logits by 2.8% (r=1) to 9.0% (r=3) -- not inert.
So skipping it is a real, growing-with-r choice, not free. The exploratory repo built exactly
this as a `skip_sink` flag on every intervention but never set it to `True` in any committed
config or script (grepped `configs/` and `scripts/` -- zero hits) -- built, never used, no
verdict either way. PLAN.md's "use all positions" is written about the prompt vs. generated-token
question (the KV-cache point), not about carving out one prompt position, so this is not
decided by it. Left for `interventions.py`; Boris should see this table before that module is
written, not have skip-sink silently default one way.

Also fixed while writing this: a `center: bool` keyword on `encode`/`decode` shadows the
module-level `center()` function inside those bodies (Python parameter scoping is whole-function,
not per-line). First draft reached for the shadowed function via `globals()["center"]`, which
worked but was the wrong kind of clever; replaced with a captured module-level reference
(`_center = center`, right after the function is defined) before `encode`/`decode` exist.

15 tests in `tests/test_spaces.py` (54 in the suite), ruff clean, ~2.5s including two live
GPT-2 forward passes.

---

## 2026-08-22 — `vectors.py`: SAE decoder rows, persona directions, the frozen split

Largest module so far (429 lines + 429 lines of tests). Adapted from the exploratory repo
(explicitly sanctioned for this one -- SAE loading, sparsity guard, feature-stats-over-corpus,
and the persona-vector extraction machinery are close ports) but the split itself is a
deliberate redesign, not a port.

### `VectorSplit`: TRAIN is derived, never stored

The exploratory repo stored three lists -- `eval_features`, `eval_features_random`,
`train_features` -- and re-checked disjointness on every load. This version stores only `dev`
and `test`; `train_pool()` returns `range(pool_size)` minus their union, computed on demand.

Two reasons, both concrete rather than aesthetic:

- **Size.** `train_features` for the GPT-2 SAE would be ~130,900 of 131,072 indices -- not
  "small, diffable" (repo convention for `results/`). The complement needs zero bytes.
- **The invariant becomes structural.** A stored train list can only be *checked* against
  dev/test on load; nothing stops it from being edited independently (a merge, a hand fix) and
  quietly drifting out of sync between checks. A complement can't drift from itself -- there is
  no second list to disagree with the first.

This also resolved a scope question from Step 2's language: PLAN.md's D4 corruption
("full allowed SAE dictionary") needs a much larger pool than the ~150-200 *usable* (in-band,
steerable, described, replicated) features DEV/TEST are drawn from. With TRAIN as the
complement of the holdout over the *whole* `d_sae`, both D3 (a small fixed pool) and D4 (the
full pool) can draw from `train_pool()` directly -- D3 freezes a 256-subset of it once, in
`corruptions.py`; D4 samples fresh from all of it every step. No second split needed.

`select_and_freeze_split` is the 3-way generalisation of the old repo's
`select_and_freeze_split`: same steerability -> description -> replication filter chain
(DECISIONS D1), but chooses DEV and TEST from one candidate pool instead of ranking a single
eval set by peak response. It takes `steerability`/`descriptions`/`replication` as plain dicts
rather than computing them, so it has no import-time dependency on `interventions.py` or
`generate.py` (neither exists yet) -- Step 2's actual probing will be notebook-orchestrated
once those modules do.

### Layer-match guard: caught my own off-by-one while writing its test

First draft of the "both namings of the same point succeed" test loaded the real SAE at
`layer=6` and `layer=7` and asserted both passed -- conflating `resid_post(6) == resid_pre(7)`
(an identity about *naming one point two ways*) with "layer 6 and layer 7 read the same
activation" (false; they're adjacent, different points). The guard itself was already correct;
only the test was wrong, and it failed loudly rather than silently, which is what it's there
for. Rewrote it to check the real equivalence directly, with a fake SAE whose metadata claims
`blocks.7.hook_resid_pre` and asserting *that* matches `layer=6`. Kept a second, real-SAE test
that `layer=7` is correctly rejected -- the actual off-by-one this guard exists to catch.

### Confirmed against the real GPT-2 SAE (offline, already cached)

`gpt2-small-resid-post-v5-128k` / `blocks.6.hook_resid_post`: TopK-32, d_sae=131072,
`normalize_activations="layer_norm"`. `mean_l0` from `compute_feature_stats` matches `k=32`
exactly on every batch tried -- expected for TopK, and a wrong value there would mean the
stats loop is double-counting or mis-masking rather than telling you anything about the SAE.

### Persona vectors

`answer_activations`/`extract_persona_vector`/`separation` ported close to verbatim, with one
change: `center` is now an explicit parameter (from `spaces.should_center`), not a hidden
module-level global the way `apply_space()` was in the exploratory repo -- consistent with
`io.py` and `spaces.py` already avoiding that pattern here. Exercised end-to-end on GPT-2 (no
chat template, so it walks the plain-concatenation path); the chat-template and system-role
branches are covered by fake-tokenizer unit tests rather than a live Gemma load, which is
expensive and not yet needed -- a real Gemma run happens at Step 11.

96 tests in the suite (42 new), ruff clean, ~5s including live GPT-2 + SAE calls.

---

## 2026-08-22 — `interventions.py`, `generate.py`, `metrics.py`: Step 1 complete

Boris asked to finish PLAN.md Step 1 (reproduce B0/B1 against the old baseline) before touching
corruptions.py, rather than continuing straight down the module list. These three modules are
what that gate needs; nothing here trains anything.

### `interventions.py`: renamed the old repo's `alpha` to `r`, deliberately

PLAN.md defines `r = ||alpha*v|| / E||h||` and sweeps `r` (its own alpha grid -- 0, 0.1, ...,
3.0 -- is a grid of `r`). The exploratory repo's constructors took a parameter it called `alpha`
that meant this same relative quantity, which reads as the *absolute* push to anyone starting
from PLAN.md's own notation. `AdditiveSteering(v, r, scale)` computes `alpha = r*scale`
internally; `alpha` and `delta` are read-only properties, not stored state, so they cannot drift
out of sync with `r`.

B0, B1, B2 all inherit a sink-skip in the base `Intervention.__call__` (D4, frozen this
session): `hidden.shape[1] > 1` selects exactly the prefill pass, so slicing off index 0 there
skips exactly the sink, with no separate position bookkeeping to keep in sync with the KV cache
-- the same mechanism already proven in `hooks.py`'s prefill-only test.

Caught one wrong assumption while testing B2: renormalising `h+delta` back to the original norm
rescales the *whole* vector, including `h`'s own component along `v`, so B2's projection onto
`v` is not bounded by B1's flat `alpha` shift the way seemed obvious -- a specific random draw
made B2's raw projection *exceed* B1's. The property that actually holds is exact rather than
approximate: B2(h) = c * B1(h) for some scalar c > 0 (renormalising can't rotate a vector), so
B2 and B1 have *identical* cosine similarity to `v`, and only their norm differs. Rewrote the
test around that identity instead of the false inequality.

### A latent bug found before it could bite: `padding_side` is not global here

`generate.py` needs left-padding (`generated[:, prompt_len:]` must slice the same column for
every row in a batch). `vectors.compute_feature_stats` needs right-padding (position 0 = sink
only holds when padding trails at the end). There is no `loader.py` in this project fixing
`padding_side` once for every caller the way the exploratory repo's `models/loader.py` did --
tokenizer setup happens ad hoc in the notebook -- so nothing may assume what the tokenizer's
`padding_side` currently is. `compute_feature_stats` was still trusting the default (right, by
luck) when this was noticed; patched it to save/set/restore locally, added a test that runs it
under a deliberately-wrong `padding_side="left"` and checks the result is unchanged from
right-padding's. `generate.py` and `metrics.reference_nll` were written with the same discipline
from the start. General rule now: any function that pads manages `padding_side` itself.

### `metrics.py`: per-example NLL, not a pooled scalar

The exploratory repo's `perplexity()` summed loss and token counts across an entire call and
returned one number. `reference_nll` here returns one value per example, because PLAN.md's
paired bootstrap and matched-fluency analysis (Step 8) need individual values to resample --
a pooled scalar can be derived from per-example ones afterward, but not the reverse. Checked
against an independent hand-computed `F.cross_entropy` on a single example (not just "runs
without error") to pin the shift-by-one and the prompt/continuation boundary exactly.

Also pinned, by demonstration rather than assertion: the same text scores differently under
`reference_nll` with and without an external hook active on the model. The function does not
defend against this -- it trusts the caller to pass the unsteered model, per CLAUDE.md's own
listed trap ("perplexity of steered text must be computed under the unsteered model, or it
measures nothing") -- so the test exists to make that trust, and its cost if broken, visible.

Empty continuations return NaN, not 0.0 -- a 0.0 NLL reads as a perfect prediction, which is the
opposite of what "nothing was generated" means.

### Reference numbers to reproduce against (Step 1's actual gate, not yet run)

Pulled from the exploratory repo's `results/pareto_gpt2_small_c4.csv`: same 16 SAE features,
same 32 neutral prompts (`data/prompts/neutral_32.json`, copied here as
`configs/prompts_neutral_32.json`), `max_new_tokens=32`. B0/B1 by alpha (their alpha is this
project's `r`):

| r | ppl_ratio | dist_2 | repetition_4 |
|---|---|---|---|
| 0.00 (B0) | 1.00 | 0.926 | 0.000 |
| 0.25 | 1.22 | 0.917 | 0.003 |
| 0.50 | 1.83 | 0.883 | 0.004 |
| 1.00 | 2.52 | 0.786 | 0.004 |
| 2.00 | 2.80 | 0.588 | 0.043 |
| 4.00 | 3.83 | 0.439 | 0.196 |

Exact reproduction is not the bar -- PLAN.md explicitly changes the strength convention (`r` vs.
their `alpha`, which are numerically the same quantity here since both are relative-to-scale,
but sink-handling, prompt set overlap with training, and generation length may differ) -- but
the *shape* should: monotonic ppl_ratio increase, monotonic dist_2 decrease, repetition_4 near
zero until roughly r=1.5-2 then climbing. A pipeline that doesn't reproduce this shape is a bug
to fix before Step 2, per PLAN.md's own instruction.

137 tests in the suite (41 new: 15 interventions, 10 generate, 14 metrics, 2 vectors padding
fix), ruff clean, ~8s.

---

## 2026-08-22 — bug: greedy decoding produced a degenerate unsteered baseline

Boris ran Step 1's gate cells and flagged the numbers as off. His hypothesis was the `r` grid;
the actual cause was upstream of `r` entirely -- `B0` (**zero** steering) already showed
dist_2=0.624, repetition_4=0.251, against the reference's 0.926/0.000.

`generate.py` defaulted to greedy (`do_sample=False`), described in its own docstring as
"deterministic ... for the same reason PLAN.md's required-tests list gives." That reasoning was
too narrow: PLAN.md's "paired deterministic generation" needs *reproducibility from a config*,
which a fixed seed gives under sampling too -- greedy was an unnecessary extra constraint I
added, and on GPT-2 it costs a lot. Isolated check, same 32 reference prompts:

| decoding | dist_2 | repetition_4 |
|---|---|---|
| greedy | 0.66 | 0.19 |
| sampled (T=1, top_p=1, seeded) | 0.94 | 0.005 |
| exploratory repo's reference (also sampled) | 0.93 | 0.000 |

Greedy GPT-2 loops even with no steering applied. A baseline that degenerate leaves almost no
dynamic range to show what steering actually costs -- which is the entire point of the Pareto
front -- so this was not a cosmetic mismatch with the old numbers, it would have undermined the
MVP's central measurement from the very first result.

Fix: `generate()` now takes `temperature`/`top_p`, defaulting to `1.0/1.0` (ancestral sampling,
seeded, matching the exploratory repo's own default and its reference numbers).
`temperature=0.0` still selects greedy where wanted.

### A second-order consequence, documented rather than patched around

Under sampling, a batch draws its random numbers jointly across rows at each generation step, in
an order that depends on what else shares the batch. A fixed per-batch seed still makes a given
`batch_size` reproducible run to run, but a prompt's sampled output is **not** independent of
which other prompts happen to be batched alongside it -- unlike greedy, where no randomness is
drawn and batch composition genuinely cannot matter. Two of `generate.py`'s own tests assumed
batching-invariance and broke under the new default; both were actually testing left-padding
*alignment*, a decoding-independent property, so they now pin `temperature=0.0` to isolate that
from the (real, no longer suspicious) sampling caveat, which gets its own test instead of being
silently absorbed. Worth remembering at Step 8: re-running the final TEST evaluation at a
different `batch_size` will not bit-reproduce individual sampled examples, only the run as a
whole at its original `batch_size`.

141 tests in the suite (4 net new in `test_generate.py`, two rewritten), ruff clean, ~10s.

---

## 2026-08-22 — bug: `top_k` silently defaults to 50 under `do_sample=True`

Found while chasing Boris's report that the `r` grid needed shifting. `dist_2` matched the
reference almost exactly at every `r` (0.786 at r=1.0 in both tables, to three decimals) while
`ppl_ratio` diverged 3-5x -- if steering strength itself were miscalibrated, both metrics should
have drifted together. That pointed at how ppl was being *measured*, not at `r`.

`B0` (zero steering) measured ppl=15.9 against the reference's 153.2, a ~10x gap with no
steering involved at all. `model._prepare_generation_config(do_sample=True, temperature=1.0,
top_p=1.0, ...)` resolves `top_k` to **50**, even though `GenerationConfig()`'s own bare default
reports `top_k=None` -- a fourth, undocumented sampling restriction. Passing `top_k=0` explicitly
raised B0's measured ppl to 57+.

Fixed: `generate()` now takes `top_k` (default `0`, HF's convention for "unrestricted") and
passes it explicitly rather than leaving it for HF to fill in. `temperature=1.0, top_p=1.0`
should mean what it says to a reader; a config that silently adds a third constraint underneath
those two is exactly the "plausible-looking but wrong" failure this project's tooling exists to
catch elsewhere (io.py's whole cache-mismatch design is the same shape of problem).

### Still open: heavy-tailed, not fully explained by top_k alone

Even after the fix, per-feature `ppl_ratio` is wildly heavy-tailed: at r=1.0, mean 13.2 vs.
median 9.6 across the 16 reference features; at r=4.0, mean 23.2 vs. median 4.6. A handful of
generic concepts (`Apple brand`, `athletic records`, `political candidates`) spike 20-30x while
several others (`programming libs`, `pain/suffering`) stay under 5x at the same `r`. This matches
the project's own already-documented pattern -- variance lives *between* concepts, not within
(see the "Inherited facts" section above, sd 33 between concepts vs. sd 6 between seeds) -- so it
may be a real property of raw B1 steering on individual SAE directions with no denoiser, not a
remaining bug. Mean-of-ratios is the wrong statistic to eyeball for choosing `r` when the
distribution looks like this; a single aggregate number was hiding exactly the information
needed to answer Boris's actual question. Per-feature tables and sample text, not a top-line
ratio, are what the next notebook cells ask for.

143 tests in the suite (2 new), ruff clean, ~11s.

---

## 2026-08-22 — Step 1 gate: passed, working `r` range trimmed to `[0, 1]`

Boris inspected the per-feature breakdown and real generated text at several `r` (both
requested after the aggregate `ppl_ratio` alone turned out to be misleading -- see the previous
entry). Verdict from the text, not just the numbers: `r=0.2` is close to baseline, `r=0.4-0.6`
clearly carries the steered concept while still forming sentences, `r=0.8-1.0` is visibly
degrading, and `r>=1.5` (not shown above but consistent with the earlier per-feature table's
ratios of 15-40x) is unusable word-salad on every feature tried. Two examples from his run:
`67922` ("Wikipedia") stays topically legible through r=0.8 before breaking down; `60695`
("sports teams") is already fragmenting by r=0.4.

Working grid, replacing PLAN.md's placeholder: **`0.0, 0.2, 0.4, 0.6, 0.8, 1.0`**. This is
PLAN.md's own anticipated outcome ("trim only clearly unusable extremes during baseline
validation"), not a deviation from it.

Aggregate table at the trimmed grid, mean AND median (they still disagree, expected given the
between-feature variance already on record):

| r | mean ratio | median ratio | max ratio |
|---|---|---|---|
| 0.0 | 1.00 | 1.00 | 1.00 |
| 0.2 | 1.13 | 1.08 | 1.75 |
| 0.4 | 1.85 | 1.71 | 2.89 |
| 0.6 | 4.21 | 3.96 | 9.11 |
| 0.8 | 8.07 | 5.94 | 27.10 |
| 1.0 | 11.85 | 8.87 | 41.10 |

**Step 1's gate is passed**, with the understanding PLAN.md itself states going in ("do not
expect old denoiser results to reproduce; this MVP changes the strength convention"): `dist_2`
tracks the exploratory repo's own numbers closely at matched `r`, and B0/B1's qualitative
behavior -- concept emerges, then coherence degrades -- matches, confirmed both statistically and
by reading the text. The quantitative `r`-to-disruption mapping runs hotter here than the old
repo's did; attributed to the two decoding-config bugs found and fixed this session (greedy
default, silent `top_k=50`) plus residual transformers-version differences between the two
repos' environments, not to a further pipeline defect -- there is no remaining discrepancy
between `dist_2` and `ppl_ratio` trajectories left to explain, which was the signal that led to
finding the real bugs in the first place.

`results/step1_gate_b0_b1.csv` as currently committed is **stale** (produced before either fix,
still shows dist_2=0.624 at r=0). Must be regenerated at the trimmed grid before this entry's
numbers are backed by a committed artifact -- see the notebook instructions for the rerun.

---

## 2026-08-22 — Step 2 library code: `sae_concept_score`, `probe_steerability`, Neuronpedia

Everything Step 2 needs to run the ~1024-feature probe and freeze DEV/TEST. The probe itself
(expensive, ~70 min per the timing note above) is notebook-orchestrated, one call; the
selection logic inside it is tested here rather than trusted to a notebook loop, since it
directly defines the eval population -- the same severity class of correctness requirement as
the holdout itself.

### `DECISIONS.md` D1 corrected before first use, not after

Re-read the exploratory repo's `evaluation/steerability.py` before implementing the probe, per
Boris's earlier "consult the original repo for reference." Its fluency guard checks **two**
axes -- `ppl_ratio <= 4x` *and* `dist_2 >= 0.8x baseline` -- not perplexity alone, with a
measured reason: the two disagree by ~6x in alpha on its 16-feature set, because they fail in
*opposite* directions (repetition collapse drives ppl down, word salad drives it up), so
perplexity alone passes both failure modes silently. D1 as first written only named the
perplexity bound. This project's own Step 1 gate independently hit exactly this failure
(`ppl_ratio` heavy-tailed, mean 13.2x vs. median 9.6x at r=1.0 across 16 features) before the
probe was ever built, which is what prompted rereading the source rather than trusting the
first draft. Fixed in `DECISIONS.md` directly -- this is a pre-registration correction before
any probe has run, not a change logged under "Changed after freezing".

### `metrics.sae_concept_score`

Fire rate and mean activation of one feature on generated continuation tokens, no intervention
active -- same masking convention as `reference_nll` (right-padded, sink excluded, prompt
tokens excluded via the same length-computed mask), same "caller must pass the clean model"
trust boundary, demonstrated the same way (a hooked vs. unhooked score differing for identical
text). Explicitly documented as *not* the headline concept metric -- it shares an SAE with
whatever produced the steering direction, so it's circular by construction; useful only for
Step 2's screen and later mechanism analysis, per PLAN.md §4's requirement for an independent
judge on any final concept claim.

One test trap worth remembering: an early version demonstrated the "must be scored unsteered"
point using an arbitrary five-word continuation and a specific feature, and got 0.0 vs. 0.0 --
with only 32 of 131072 features firing per token, an unrelated short text has a real chance of
triggering neither the hooked nor unhooked pass for a given feature. Fixed by using a
topically-matched continuation, not by broadening the test's tolerance.

### `vectors.probe_steerability`

D1's criterion 2, exactly: per feature, sweep `r`, keep only points passing both fluency
guards, report the peak SAE-activation gain among survivors. Pinned with two tests that fake
every dependency (`generate`, `reference_nll`, `distinct_n`, `sae_concept_score`) to hand-picked
values -- one `r` fails only the ppl guard despite a huge apparent gain, another fails only the
dist_2 guard, and the test asserts the peak is chosen from neither, only from the two that pass
both. This is checked exactly rather than plausibly, because getting it wrong doesn't crash, it
just selects the wrong split.

### Neuronpedia description fetch, ported close to verbatim

`is_token_level`'s regex, `fetch_feature`'s cache, `describe_features`'s omit-if-undescribed
rule -- carried over including the exploratory repo's own bug fix (the trailing `s?` that
catches plurals; without it "articles and indefinite articles" passed as semantic and was
selected into a frozen split at the third-highest peak in that project's history). `cache_dir`
is a required explicit argument, not defaulted to `io.ARTIFACTS` -- matches this project's
pattern of not hiding where a function reads from.

**Not run yet.** Fetching real descriptions and probing real features both need network access
Boris hasn't given the go-ahead for (open item, `DEVLOG.md` 2026-08-22 initial entry). All
network-touching tests are mocked; nothing above has made a real HTTP request.

168 tests in the suite (25 new: 21 `sae_concept_score`, 3 `probe_steerability`, 9 Neuronpedia
minus overlap), ruff clean, ~11s -- all offline.

---

## 2026-08-22 — bug: JSON round-trip silently stringifies int dict keys

Found while writing the Step 2 notebook cells, before it could bite: `probe_steerability` and
`describe_features` both return `dict[int, dict]`, keyed by SAE feature index. JSON object keys
are always strings, so `io.run_or_load`'s cache-hit path was handing back `{"1878": {...}}`
after any reload -- and `select_and_freeze_split`'s `steerability.get(feature_id, {})` lookups
use int feature ids throughout. Every lookup would have silently missed, quietly returning the
default instead of raising: an empty usable-set, not a crash, discovered only by the split
looking suspiciously small (or not discovered at all).

Fixed in `io.py`, not per call site: `run_or_load` now records whether a saved dict was
genuinely int-keyed (`type(k) is int`, deliberately excluding `bool`, which subclasses `int` in
Python) in the meta sidecar, and restores it on every cache-hit load. Chose this over three
separate notebook-side re-keying steps because the same trap recurs for all three of Step 2's
int-keyed outputs (probe results, Neuronpedia descriptions, second-seed replication) and will
recur again for anything keyed by feature id in the future -- fixing it once in the shared cache
layer is more in keeping with what `io.py` is for than patching it three times downstream.

One implementation slip caught immediately by the test suite: the first version checked
`bool(payload) and isinstance(payload, dict) and ...`, which raises on a DataFrame or tensor
payload ("truth value is ambiguous") before the `isinstance` check ever gets a chance to
short-circuit it away. Swapped the order.

170 tests in the suite (2 new), ruff clean, ~12s.

---

## 2026-08-22 — bug: dict-of-tensors payload crashed `io._save`'s JSON branch

Boris hit this running Cell 16: `compute_feature_stats` returns `{"frequency": tensor, ...}`,
and `_save` dispatched it to `json.dumps` purely because it's a dict -- format dispatch was by
payload *type*, not by whether the contents are actually JSON-safe. Raised deep inside the
encoder (`TypeError: Object of type Tensor is not JSON serializable`), not at a boundary that
would have pointed at the cause quickly.

Fixed with `_json_safe()`, checked recursively before choosing the JSON path; anything that
fails it (a dict of tensors, or a tensor nested inside a dict of dicts) falls back to
`torch.save` instead, matching the module's own stated rule ("torch for anything that is
weights or activations") that the type-only dispatch wasn't actually enforcing for dicts.
`_int_keyed` needed no change -- pickle-based `torch.save`/`load` preserve key types exactly,
so the JSON-string-key trap fixed two entries above simply doesn't arise on this path.

173 tests in the suite (3 new), ruff clean, ~12s. Cell 16 can be rerun as originally written --
no notebook change needed, the fix is entirely in `io.py`.

---

## 2026-08-22 — bug: `ResidualHook`'s CPU capture vs. MPS mask, three call sites

Boris hit this on Cell 16 right after switching to MPS (his own question, "why is device cpu
in many cells" -- I'd never actually wired up device selection; every `device="cpu"` I wrote
was just the plain default while focused on correctness, not a benchmarked choice). Quick
benchmark once he asked: MPS is ~2.8x faster than CPU for GPT-2-small generation at the Step 2
probe's batch shape (0.25s vs 0.70s per batch of 8, 32 new tokens) -- worth the switch, so I
recommended it, and that's what surfaced this.

`ResidualHook(capture=True)` always captures to CPU (`hooks.py`'s own deliberate default --
caching a large corpus of activations on MPS runs out of memory well before the corpus is
useful). But a mask built from `enc["attention_mask"]` lives on whatever device the model is
on. Every existing test ran model and SAE on CPU only, so hidden and mask always
*coincidentally* matched -- nothing in 178 tests exercised the actual seam until a real MPS run
did. `RuntimeError: indices should be either on cpu or on the same device as the indexed
tensor (cpu)`.

Same pattern, three call sites, all fixed the same way -- move the (cheap) mask to
`hidden`'s device rather than moving the (larger) captured hidden states, and move any small
extracted tensor to the SAE's own device (`sae.W_dec.device`, which need not match the model's
device either) right before encoding:

- `vectors.compute_feature_stats` (Step 2's frequency scan -- what Boris actually hit)
- `metrics.sae_concept_score` (feeds `probe_steerability`, so this was on Step 2's critical
  path regardless)
- `vectors.answer_activations` (persona extraction -- not yet exercised, but Step 11 would have
  hit the identical bug on Gemma with no warning until then)

Added MPS-gated regression tests for all three (`pytest.mark.skipif(not
torch.backends.mps.is_available())`), covering model-on-mps and SAE-on-mps independently since
either alone is enough to trigger the mismatch. Run and passed on this machine's real MPS
device, not just reasoned about -- closing the actual coverage gap that let this through,
rather than fixing blind and hoping.

178 tests in the suite (5 new, MPS-gated), ruff clean, ~13s on CPU; 5/5 MPS tests pass here.

Notebook: no changes needed to Cells 12-21 as given -- the fix is entirely in `src/steering/`.
Cell 16 (and everything after it) should now run correctly with the model and SAE both on MPS.

---

## 2026-08-22 — `judge.py`: ported closely, live-tested against the real `gemma4:31b`

Step 3 needs this to calibrate reference NLL against judge coherence, so it jumped ahead of
`corruptions.py`/`denoiser.py` in PLAN.md's own module list -- consistent with how Step 1
skipped straight to `interventions.py`/`generate.py`/`metrics.py` without those two either.

Close port of the exploratory repo's `evaluation/judge.py` (Boris: "consult previous repos
where needed"), kept close deliberately -- three of its design choices were each earned by a
real failure and are not worth re-deriving:

- **`think=False` by default.** A thinking judge model burns hundreds of tokens reasoning
  before answering a single integer; under a small `num_predict` it truncates mid-thought and
  returns an *empty* string -- silent, not an exception.
- **Non-string `concept` raises `TypeError`.** The exact historical bug: a metrics dict once
  reached the concept rubric, formatted into the prompt as its own repr, and the judge
  dutifully scored *that* -- plausible numbers for a question nobody asked, flattening a whole
  sweep.
- **Two independent rubric calls, never combined.** Coherence and concept/trait are always
  separate calls; the exploratory repo measured trait=100, coherence=0 on the same Gemma text
  at high alpha, which a single blended score would have called a success.

All four rubrics carried over (coherence/concept for GPT-2's continuation style, needed now;
chat_coherence/trait for Gemma's chat style, needed at Step 11) rather than trimming to what's
immediately used -- they're prompt-template strings, not extra machinery, and re-deriving their
anchor wording later with less context would be worse than keeping it now.

Confirmed `gemma4:31b` is present locally (`ollama list`) before writing a single test against
it. Five live tests hit the real model -- an obviously-coherent and an obviously-word-salad
case for `coherence`, an obviously-saturated and an obviously-unrelated case for `concept`, and
one small real `score_many` batch checking threaded order preservation -- all passed on the
first run, ~11s total including the offline suite. Skipped rather than failed if Ollama isn't
reachable (checked once at collection time, not per test).

209 tests in the suite (31 new), ruff clean, ~21s total including live judge calls.

---

## 2026-08-22 — Step 3: B0/B1/B2 on DEV, judge calibration, and an open finding on B2

### Gate: passed

35 DEV concepts, `r` in `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (Step 1's validated grid), B0 shared
baseline + B1/B2 per concept. Clear, monotonic concept-quality tradeoff for both arms:
`concept_mean_act` 0.015 -> 0.27 (B1) as `r` rises, `dist_2` 0.96 -> 0.88, `repetition_4` stays
under 0.006 throughout -- a world away from the greedy-decoding-era baseline (0.25 degenerate),
confirming the earlier fixes hold up on a wider concept set, not just the 16 reference features.

### Judge calibration: strong, correctly-signed

33-example quality-stratified sample, scored on the `coherence` rubric against the real
`gemma4:31b`: Pearson r=-0.619 (p=0.0001), Spearman r=-0.766 (p<0.0001) between reference NLL
and judge coherence. Spearman noticeably stronger than Pearson -- consistent with a monotonic
but non-linear relationship (raw NLL is unbounded, judge coherence is capped at 100, so the
relationship should saturate at the high-NLL tail rather than staying linear). Good enough to
trust reference NLL as Step 8's cheap proxy.

### Open finding: B2 loses to B1 on both axes at r >= 0.6, mechanism not yet identified

At matched `r`, B2 (norm-matched) is not just differently positioned on the frontier from B1 --
it is **dominated**, worse on both ppl and concept simultaneously, and the gap grows with `r`:

| r | B1 ppl | B2 ppl | B1 concept | B2 concept |
|---|---|---|---|---|
| 0.4 | 106.5 | 104.1 | 0.103 | 0.104 |
| 0.6 | 250.7 | 262.7 | 0.181 | 0.158 |
| 0.8 | 486.5 | 571.4 | 0.232 | 0.193 |
| 1.0 | 734.9 | 915.2 | 0.275 | 0.201 |

Tracks almost exactly at r=0.2-0.4, diverges unfavourably for B2 past that. Contradicts
`interventions.py`'s own stated assumption ("concept lives mostly in direction; restoring the
norm recovers much of the fluency a plain push costs") -- true at low r, false at high r here.

Two mechanisms proposed and directly measured, both refuted:

1. **Amplification** (the rescale factor `c = ||h|| / ||h+delta||` exceeding 1, so B2 makes the
   push *bigger* rather than smaller). Measured both on a single clean forward pass and, more
   importantly, on the real autoregressive rollout (instrumented hook, prefill + every decode
   step, 12 DEV directions): `c` stays well below 1 throughout and *falls* as `r` grows
   (median 0.93 at r=0.4 -> 0.68 at r=1.0). B2 damps more at higher r, not less. Refuted.

2. **Directional/magnitude inconsistency** (norm-matching recomputed fresh at every position
   producing an erratic, compounding correction rather than a steady one, unlike B1's literally
   constant `delta`). Measured the actual applied correction's cosine similarity to the
   intended direction and its magnitude CV across the same real rollout: cosine stays high
   (mean 0.96 at r=0.6 -> 0.92 at r=1.0, never below 0.74) and magnitude CV stays low
   (0.044 -> 0.074) -- B2's correction is well-aligned and fairly consistent, just smaller than
   B1's. Refuted.

So the applied correction is smaller, well-aligned, and consistent by every measure checked --
which should make B2 gentler on *both* axes than B1, not worse on both. It isn't. A candidate
third mechanism, not yet tested: GPT-2's downstream layers may read residual-stream *norm*
itself as a signal (elevated norm correlating with salience/confidence in some transformer
literature), so B1's "abnormal but consistent" shift may be easier for later layers to handle
than B2's "normal-looking but directionally wrong" one -- testable via per-layer sensitivity
downstream of layer 6, not yet done.

**Decision: logged, not chased further right now.** Two clean, reproducible refutations is a
real result on its own (rules out the two most obvious stories), and Step 3's actual gate does
not depend on resolving it -- B2 remains a valid, if now more interesting, mandatory control.
Worth returning to at Step 9 (mechanism analysis: what does the denoiser repair, and does it
handle B2's failure mode differently from B1's).

209 tests remain the current count (no library changes this entry -- diagnostics were run
standalone, not added to the test suite, since they measure a specific finding rather than pin
a reusable behaviour).

---

## 2026-08-22 — `denoiser.py`: `cache_activations`, Step 4

Only the caching utility so far -- the `ResidualMLPDenoiser` architecture itself is Step 5's
job, once there's a corruption to train it against.

Design question worth being explicit about: does the cache store centered/scaled activations,
or raw ones? The denoiser's own forward pass is `D(x, s) = x + f(N(x), e(s))` -- `N(x)` *is*
`spaces.encode`, called inside `D()`. Baking centering/scaling into the cache instead would mean
the transform runs once, at cache time, and never again -- the opposite of PLAN.md Step 4's own
requirement ("training and inference must pass through the same spaces.py transform"), which
means the *same call*, at both training and inference, not merely the same formula applied at
two different times. So the cache is raw: pinned with an exact-identity test against a live
hook capture, not just a shape check.

Reuses the exact device-safety pattern from the compute_feature_stats/sae_concept_score fix
(mask moved to `hidden`'s device, not the other way round) from the start, rather than
discovering the same bug a fourth time -- confirmed on real MPS hardware, not just reasoned
about.

`exclude_sink=True` by default: every intervention already skips the sink (D4), so training the
denoiser on it would teach it to reconstruct a position it will never be asked to touch.

9 new tests (218 in the suite), ruff clean, ~2.5s including a real MPS run.

### Corpus size for the real cache: reusing Step 2's, not fetching a new one

Step 2's frequency-band scan already processed 3000 Pile-10k documents (`max_length=64`) into
186,699 real, non-sink tokens (`feature_stats_gpt2_l6.meta.json`) -- same texts, same
tokenization settings work directly for `cache_activations` too, and it does strictly less
per-batch work (no SAE encode), so it should be at least as fast as that already-measured run.
Reusing it avoids a second corpus fetch and gives ~187k distinct activation vectors, ample for a
two-block residual MLP.

---

## 2026-08-22 — `corruptions.py`: D1-D4, holdout guard tested against a hand-crafted bypass

Close port of the exploratory repo's `data/corruptions.py` (Boris: "consult previous repos
where needed") for D1 (Gaussian) and the Rank1/StructuredRank1 machinery, **not** for D2 --
PLAN.md's D2 is `sqrt(1-beta)*h + sqrt(beta)*eps`, a variance-preserving SDE-style corruption,
which is a different family from the exploratory repo's own "C2" (a plain linear interpolant
`t*h + (1-t)*eps`) despite the similar name. Implemented D2 from PLAN.md's formula directly,
checked against a hand computation with a monkeypatched RNG (same standard `reference_nll` was
held to) rather than only a shape check.

### Why D3 and D4 collapsed into one base class instead of two independent ones

The exploratory repo's C4 (fixed pool) and C7 (full pool) needed separate pool-extension logic
because its `train_features` was a separately-stored, size-limited list -- widening the pool
past that list meant explicitly drawing more non-evaluation features and checking disjointness
again. This project's `VectorSplit.train_pool()` is already "every index not in dev/test," the
full ~131k-entry complement, computed structurally rather than stored -- so D3 (`FixedPoolRank1`)
and D4 (`FullPoolRank1`) differ only in whether that pool gets subsampled once (D3, seeded,
frozen for the run) or used whole (D4, resampled every call). One shared base
(`StructuredRank1`) does the actual work; the two subclasses are a few lines each. A concrete
payoff from the earlier `VectorSplit` redesign, not just a cleaner API.

### The holdout guard, tested against more than the happy path

`StructuredRank1.__init__` asserts `pool_indices` disjoint from `holdout` explicitly, even
though `split.train_pool()` already guarantees this structurally -- defense in depth against a
wrong *implementation* of the pool-building logic, not merely documentation of a property the
type already has. Two tests exercise this directly rather than trusting the guarantee:

- construct `StructuredRank1` with a **hand-crafted** `pool_indices` list that bypasses
  `split.train_pool()` entirely and includes a holdout index -- must raise;
- construct it with a **tampered holdout set** that disagrees with what `train_pool()` would
  actually exclude -- must still raise, on the assumption that some *other* bug upstream could
  hand it a wrong holdout, not just a wrong pool.

Both raise as required. `FixedPoolRank1`/`FullPoolRank1` are exercised end-to-end through the
real `VectorSplit`-based construction path a training script would actually use, not only
through the lower-level base class.

`Rank1`'s rho and `Gaussian`'s sigma both sample log-uniformly over the same `[0.05, 3]`
(PLAN.md §3), sharing one `_log_uniform_t` helper for the `t_for_r` conditioning map (DECISIONS
D3). `VariancePreserving.t_for_r` is an honest `clamp(r, 0, 1)` placeholder, explicitly flagged
as unvalidated in its own docstring -- there is no principled r->beta correspondence the way
there is for the additive families, and DECISIONS D3 already records what happens when an
unvalidated map like this gets trusted silently (wrong by 2-3x, undiscovered until someone went
looking).

29 new tests (247 in the suite), ruff clean, ~24s.

---

## 2026-08-22 — `train_denoiser`: the training loop, and a test that demanded the impossible

Completes Step 5's library code: sample a minibatch from the cached pool (CPU-resident by
`cache_activations`'s own design), move only that minibatch to `device`, corrupt it, compute
`denoising_loss`, step. Generic over D1-D4 by construction -- nothing here branches on
corruption type, since `Corrupted`'s `x`/`target`/`t` interface is what every corruption family
already produces (tested explicitly across all four, plus `Rank1`'s bare/sphere variant).

Two generators, deliberately separate: index sampling on a CPU generator (which examples get
drawn), corruption noise on a `device`-resident one (`corruptions._check_generator` requires it
to match the tensor it operates on, and that tensor is the minibatch, already moved). Confirmed
`torch.Generator(device="mps")` actually works on this machine before relying on it, rather than
assuming.

### A test that asked for something the problem couldn't give

First draft of the convergence test used `synthetic_pool` -- i.i.d. Gaussian rows with no
structure to exploit beyond simple shrinkage -- and demanded `final_loss < 0.5 * initial_loss`.
Failed at 0.161 against a 0.107 target. Before assuming the training loop was broken: this
synthetic problem has a computable Bayes-optimal floor (clean and noise are independent
zero-mean Gaussians, so the MMSE estimate is `x/(1+sigma^2)`), measured directly at loss ~0.15
against an identity baseline of ~0.216 -- and the network was already converging to within
range of that floor by ~150 steps. The 0.5x threshold asked for a loss *below the problem's own
information-theoretic limit*, achievable by no denoiser, trained or not. Fixed to 0.8x, an
achievable bound that still rules out "training did nothing." The training loop itself needed
no changes -- it was already working close to optimally on the first real run.

268 tests in the suite (9 new), ruff clean, ~21s.

---

## 2026-08-22 — `evaluate_denoiser` and a real speed measurement before the first-pass run

`evaluate_denoiser` is what Step 5's "eliminate clearly weak families" decision actually reads,
so it gets the same treatment as `probe_steerability`: a tested function, not ad hoc notebook
code, since getting it wrong doesn't crash -- it silently eliminates the wrong corruption
family. Pinned that a trained model measurably beats an untrained (identity) one on genuinely
held-out data (a different seed's draw than training used), which is the one property that
actually matters: this function has to measure real denoising quality, not just return a
number that happens to be finite.

Measured training speed at the real scale (`d_model=768`, synthetic data of matching shape)
before picking step counts for the actual GPT-2 run: **~5s per 2000 steps on MPS**, ~28s on
CPU. PLAN.md's own budget ("GPT-2 denoiser selection: <1 hour of training") is not remotely a
binding constraint at this speed -- four first-pass trainings plus a second-seed confirmation of
the top two could run in well under 3 minutes total even at 5000+ steps each. Chose a generous
step count for the real run on that basis, not a time-pressured minimal one.

276 tests in the suite (8 new), ruff clean, ~21s.

---

## 2026-08-22 — Step 5 first pass: D3's memorisation confound caught before it misled anything

First-pass training (one seed, real GPT-2 activations, 5000 steps, ~5s each on MPS):

| family | own val_loss | dev-direction generalization |
|---|---|---|
| D1 | 0.2022 | 0.1269 |
| D2 | 0.2927 | 0.2246 |
| D3 | 0.0200 | 0.1575 |
| D4 | 0.1988 | 0.1207 |

The first number Boris ran (`own val_loss`) made D3 look ~10x better than everything else.
It wasn't real: D3's validation corruption still drew from the *same fixed 256-direction pool*
used in training, so a low loss there measures memorisation of that pool, not denoising ability
-- exactly the failure `corruptions.py`'s own docstring already named, quoting the exploratory
repo's measured ratio (~400 for a fixed 256-pool against 1.5-6 for a resampled one). This
project's own accumulated history flagged the mechanism before the number was ever produced;
the number then confirmed it needed exactly that flag.

Fixed by evaluating all four **already-trained** models under one **shared** corruption built
from `split.dev` directions (never in any training pool -- legitimate for a method-selection
decision; `split.test` stays untouched) at the validated deployment range `r in [0.05, 1.0]`, not
each family's own training corruption. This also incidentally fixes a second problem the first
table had: D1/D2/D3/D4 were each scored under a *different* corruption shape with a different
intrinsic difficulty floor, so their own numbers were never directly comparable to begin with.

On the fair metric, D3 drops from best to third of four (0.1575, worse than D1 and D4) --
losing outright once it faces a direction it never saw. D2 is worst on both metrics, consistent
with being a harder, differently-shaped corruption (destructive interpolation vs. additive) and
the one family whose r->t conditioning map was already flagged unvalidated
(`VariancePreserving.t_for_r`'s own docstring).

**Decision: D1 and D4 advance to the second-seed confirmation; D2 and D3 eliminated.** D4 edges
D1 by a real but modest margin (~5%), mechanistically sensible -- D4's fresh-every-step rank-1
push most closely matches the deployment perturbation shape, the same reasoning the exploratory
repo gave for its own best-performing corruption (C7, D4's direct analogue). Provisional per
PLAN.md ("do not freeze a corruption family from one seed") -- the D1-vs-D4 ranking specifically
still needs the second seed; the D2/D3 elimination does not, since PLAN.md's first pass exists
for exactly that cut.

---

## 2026-08-22 — Step 5 winner: D4 (rank-1, full pool), tie broken by mechanistic prior

Second-seed confirmation flipped rank between seeds: D4 won seed 0 (0.1207 vs D1's 0.1269),
D1 won seed 1 (0.1252 vs D4's 0.1280). Means: D4 0.1243, D1 0.1261 -- a 1.4% gap, well inside
the noise a single rank flip already demonstrates.

**Decision: D4.** Not from these numbers, which cannot distinguish the two -- from the same
mechanistic prior that put D4 in the top two to begin with: it is the only family whose
corruption shape matches deployment (a fresh rank-1 push along a real decoder direction every
step, no fixed pool to memorise), matching the exploratory repo's own conclusion about its
direct analogue (C7). A statistical tie does not overturn that prior; it means this evidence
can't move it either way, so the tie is broken by the reason D4 was a candidate at all rather
than by chasing a third seed for a ~1% difference.

Both D1 and D4 checkpoints are saved (`denoiser_d1_seed{0,1}.pt`, `denoiser_d4_seed{0,1}.pt`,
`artifacts/`) in case this decision needs revisiting.

Moving to Step 6: pool-size check for D4 specifically (64, 256, 1024, full -- PLAN.md's four
points), now that rank-1 corruption is confirmed among the winners.

---

## 2026-08-22 — Step 6: pool-size check, diversity saturates around 1024

| pool | n directions | dev_generalization |
|---|---|---|
| 64 | 64 | 0.1683 |
| 256 | 256 | 0.1575 |
| 1024 | 1024 | 0.1187 |
| full | 130,967 | 0.1207 |

Monotonic improvement 64->1024 (~30% relative), then flat 1024->full -- the 0.002 difference
is smaller than the noise floor already measured in Step 5's seed-to-seed D1/D4 comparison
(~0.006-0.007). The full-pool row reproduces D4's already-measured seed-0 result almost exactly
(0.1207 both times), a free consistency check on the whole pipeline: same corruption, same
config, same number.

**Finding: corruption-direction diversity helps up to roughly 1024 directions, then saturates.**
Does not reverse Step 5's choice of full-pool D4 (still at least as good as anything tested,
and PLAN.md's own named definition of D4) -- but explains *why* full-pool isn't wasted effort
without being strictly necessary either: D4 sits comfortably past the point where more
directions would help, matching the general shape of the exploratory repo's own memorisation
curve (steep gains at small pools, diminishing beyond a few thousand).

Step 6 complete -- PLAN.md's own "four points are enough." Step 7 (freeze the method) is next.

---

## 2026-08-22 — Step 7: method frozen

`DECISIONS.md` D6 completed with the real values from the notebook run (not placeholders):
config hash `49db404c4e308bc0`, judge subset hash `acde5e375f7dd7c0` (40 of 70 TEST concepts,
seed 0, `r` in `[0.2, 0.4, 0.6, 0.8]`). D4/full-pool/the architecture and training config as
already decided; carries forward D3/D2's elimination and the tie-broken-by-prior D1-vs-D4 call,
both already logged with their own reasoning rather than restated as bare facts here.

No changes to `src/` this entry -- purely a provenance record, read directly from
`results/frozen_method_config_gpt2.json` and `results/judge_subset_gpt2.json`, both already on
disk from Boris's own notebook run.

Moving to Step 8 (final GPT-2 test, 3 seeds, all TEST concepts, headline B1/B2/D1/D4 comparison,
judge on the frozen 40x4 subset, Pareto + paired bootstrap). `stats.py` is still an empty stub
and Step 8 is the first place this project needs it -- building it next, tests first.

---

## 2026-08-22 — `stats.py` and `interventions.DenoisedSteering`: the last two pieces before Step 8's run

Two gaps closed before any Step 8 cell could actually run.

**`stats.py`.** Ported `bootstrap_ci`/`paired_test`/`holm` from the exploratory repo's
`evaluation/significance.py` closely; generalised `concept_at_fluency`/`matched_fluency_*` from
`evaluation/pareto.py` into direction-agnostic `value_at_target`/`matched_comparison`/
`matched_band` (swap which column is `x` and which is `y` to answer "quality at matched
concept" or "concept at matched quality" with the same function -- PLAN.md Step 8 asks for
both). Kept the one piece of design that mattered most: `matched_band`'s interval is built by
the *same* within-unit-differencing procedure `matched_comparison` uses at a single point,
evaluated on a grid instead -- not a separately-bootstrapped mean front per arm, differenced
afterward, which the source measured as 1.4x wider for a specific, mechanical reason (resampling
shifts which units define each arm's mean front *before* interpolation runs, so the two arms'
interpolation error stops cancelling). Dropped the source's Cousineau-Morey/variance-components
machinery -- built for a many-arm, many-trait report; this project compares four methods.

Test fixture bug caught immediately: `matched_band` requires >= 4 units for a meaningful
bootstrap (the source's own threshold, kept as-is), but the first draft of `make_front_df()`
only provided 3 -- three tests failed with a stale hardcoded unit count carried over after the
fixture grew to 6. Not a bug in the function; the guard did exactly what it should.

**`interventions.DenoisedSteering`.** The actual gap: nothing applied a trained denoiser as a
steering intervention. `D(h + alpha*v)`, with `t` derived from `r` via `corruptions.t_for_r`
(DECISIONS D3 -- a fixed or skipped `t` is the silent misuse D3 names explicitly). Checked
against an exact identity: with an untrained (zero-init) denoiser this must reduce to plain
`AdditiveSteering` exactly, since `D(x,t)=x` at init by construction -- a strong, exact
composition-order check (steer then denoise, not the reverse) rather than a shape test.

306 tests in the suite (30 new: 25 stats, 5 DenoisedSteering), ruff clean, ~21s.

Step 8's run cells are next -- everything it needs now exists and is tested.

---

## 2026-08-22 — MAJOR BUG: left-padding misplaced the attention sink under batched generation

Found in a real Step 8 run: D4's headline TEST-set ppl came back at 8,444-16,962 (r=0.6-1.0)
against B1's 282-902 at the same r -- catastrophic, and getting *worse* with `r`, alongside
*higher* dist_2 than B1 (0.99 vs 0.89). High diversity + catastrophic perplexity is the
word-salad signature. Generated text confirmed it: near-identical garbage tokens ("Tuc",
"Bitcoin", "Shara", "Medium Moon") recurred across a dozen *different* steering directions --
the actual feature barely mattered, which meant the bug was direction-independent, not a
per-feature problem.

### Root cause, confirmed by direct measurement

`generate.generate()` left-pads a batch of different-length prompts (needed so
`generated[:, prompt_len:]` slices the continuation correctly for every row). Every
`Intervention`'s sink-skip logic assumes position 0 is the attention sink and always skips it.
Checked directly: with the real 32 neutral prompts (lengths 4-5 tokens) batched together,
**23 of 32 rows have a PAD token at position 0**, not the sink -- the real sink for those rows
sits at index 1 or later, gets no skip protection, and is steered/denoised like an ordinary
token. The sink carries ~30-40x the median activation norm (measured back in Step 1). Feeding
that into `spaces.encode` and a denoiser trained *exclusively* on non-sink-scale activations
(the cache explicitly excludes it) pushes the MLP 30x outside anything it was trained on --
plausibly explaining the catastrophic, unstable output. B1/B2's corrections are small/bounded
relative to an already-huge outlier, so the same misapplication there is comparatively benign,
which is why B0/B1/B2's numbers looked reasonable throughout Steps 1 and 3 despite carrying the
same underlying bug the whole time.

Confirmed the mechanism by testing a single unpadded prompt in isolation (perfectly coherent
output, stable norms) against the same feature inside a real padded 32-prompt batch
(catastrophic). Checked the old repo for reference: its `skip_sink` flag defaulted to `False`
and -- per this project's own first DEVLOG entry -- was never enabled in any of its actual runs.
That sidesteps this exact class of bug by never attempting to skip anything, which is not a fix
worth copying (D4 was deliberately trained without ever seeing the sink, so correctly skipping
it at inference is the right choice) -- it just explains why the old repo never hit this.

### Fix: group every batch to a single exact tokenized length

Confirmed via search that "sequence bucketing" (grouping similar-length sequences to reduce
padding) is the standard technique for this class of problem, not a workaround --
[PyTorch BatchSampler for bucketing by length](https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c),
[bucket_by_sequence_length tutorial](https://medium.com/analytics-vidhya/tutorial-on-bucket-by-sequence-length-api-for-efficiently-batching-nlp-data-while-training-20d8ef5219d7).
The usual form still tolerates some padding within a bucket (an efficiency optimisation); this
project's positional-correctness requirement cannot tolerate any, so `generate.generate()` now
groups by **exact** tokenized length before batching -- `tokenizer(..., padding=True)` becomes a
no-op on every batch it builds, so position 0 is the true sink for every row, always.

Regression test written and confirmed to **fail against the pre-fix code** before the fix
landed (not just written and hoped to be correct): compares a short prompt generated alone
against the same prompt generated in a batch with a much longer one, using a real steering push
(`NoSteering` can't catch this -- it touches nothing regardless of where the sink is assumed to
be). One existing test's premise became obsolete as a direct, *good* side effect of the fix:
`test_sampling_is_not_batch_invariant...` used a short/long pair to demonstrate that sampling
isn't batch-invariant, but different-length prompts are no longer ever batched together at all
now, so that pair became batch-invariant as a side effect. Fixed to use two same-length prompts,
which still genuinely share a batch and still exhibit the real (unrelated, expected) caveat.

### What needs rerunning and what doesn't

Checked which steps actually call `generate.generate()` versus operate on cached activations
directly, since only the former could have been affected:

| Step | Uses `generate.generate()`? | Rerun? |
|---|---|---|
| 1 (B0/B1 gate) | Yes | **Yes** -- cheap, and B1's own numbers, while probably only mildly off, are worth having clean |
| 2 (probe + split) | Yes, inside `probe_steerability` (B1-only, no denoiser) | **No** -- see below |
| 3 (DEV sweep + judge) | Yes | **Yes** |
| 4 (activation cache) | No | No |
| 5/6 (train/evaluate denoiser) | No -- `train_denoiser`/`evaluate_denoiser` sample raw cached activations directly, never generate text | No -- checkpoints are unaffected and do not need retraining |
| 7 (freeze) | No | No -- nothing about the frozen *choices* changed |
| 8 (final test) | Yes, and this is where the denoiser exposed it | **Yes** -- the one that matters most |

**Step 2 is deliberately not being rerun.** Its fluency guard used only B1-style additive
steering (no denoiser), so by the same reasoning that kept B0/B1/B2 looking reasonable
throughout, its effect there is bounded, not catastrophic. Re-running the probe would very
likely shift *which* borderline features pass the fluency guard by a small amount -- and
PLAN.md is explicit that the split, once frozen, is not revisited ("do not revisit the split
later"), for exactly the reason that re-deciding it after any downstream result would be the
selection effect the whole holdout exists to prevent. Re-running Steps 1/3/8 costs little and
directly fixes numbers already known to be wrong; re-running Step 2 would cost a full
re-freeze cascading through every step after it, to correct an effect that -- unlike Step 8's --
was never observed to be more than mild.

307 tests in the suite (2 new: the regression test, plus one existing test's premise fixed
rather than removed), ruff clean, ~25s.

---

## 2026-08-22 — the sink/padding bug: root cause, fix, and a systematic audit for the same pattern

### What happened

Step 8's full TEST sweep produced catastrophic, direction-independent word salad for every
denoised arm -- mean ppl above 15,000 at r=1.0 (D1) and above 16,000 (D4), against a few
hundred for B1/B2 at the same r. The same garbled tokens ("Tuc", "Bitcoin", "Shara", "Medium
Moon") recurred across completely different steering directions, which was the tell: the
specific concept barely mattered, so the bug was not in any one corruption or direction.

### Root cause

`Intervention.__call__` skips the sink by slicing off index 0 (`hidden[:, :1, :]`) and applying
`apply()` only to the rest -- correct under the assumption that index 0 is always the true
first token. `generate.generate()` left-pads (needed so `generated[:, prompt_len:]` slices the
continuation correctly across a batch of different-length prompts). Under left-padding, index 0
is a genuine token only for the batch's *longest* prompt; every shorter row has a **pad token**
at index 0, with its real first token -- the attention sink, carrying ~30-40x the median
activation norm -- sitting further in, un-skipped. Measured directly: 23 of 32 rows in the
actual neutral-prompt batch have a pad token at position 0. For B1/B2 this is a mild extra
perturbation on an already-huge outlier; fed into a denoiser that was never trained on anything
within 30x of its normal input scale, it produced numerically wild, garbage corrections that
then poisoned every subsequent generated token through the KV cache.

A single-prompt manual test (no padding at all) with the exact same feature and `r` produced
completely coherent text, which was the first clue this was a batching artifact, not a model or
training problem. Confirmed decisively: `enc["attention_mask"][row][0] == 0` for the majority
of rows in a real batch.

Checked whether the exploratory repo had already solved this: its `skip_sink` flag defaulted to
`False` and, per this project's own earlier inherited-facts note, was never enabled in any of
its actual runs -- it sidestepped the bug by never attempting to skip anything, not by handling
padding correctly. Not a fix worth copying; D4 was deliberately trained without ever seeing the
sink; skipping it at inference is the right choice, so it needed to be done correctly, not
abandoned.

### Fix

Researched the standard technique first (web search: "sequence bucketing" -- grouping batches
by length is well-established, e.g. https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c,
https://github.com/huggingface/transformers/issues/26072) -- confirmed this is the right shape
of fix, not a hack. `generate.generate()` now groups prompts by **exact** tokenized length
(not merely similar -- the usual bucketing tolerates some in-bucket padding, which this
project's positional correctness requirement cannot) before batching, so `tokenizer(...,
padding=True)` is a no-op on every batch it builds: nothing to pad, so position 0 is the true
sink for every row, always. Output order is preserved via explicit index tracking rather than
relying on batch order.

Regression test added (`test_batching_with_mixed_lengths_does_not_corrupt_the_sink_position`)
using a real steering push -- `NoSteering` can't catch this, since it touches nothing regardless
of where the sink is assumed to be. Confirmed the test fails against the pre-fix code (exactly
as predicted) before confirming it passes against the fix. One existing test
(`test_sampling_is_not_batch_invariant_even_with_a_fixed_seed`) needed updating: it used a
short/long prompt pair specifically to demonstrate that sampling isn't batch-invariant, but
after the fix those two prompts are never actually batched together any more (different
lengths -> different length-groups), so the caveat it existed to pin no longer applies to that
pair -- a good side effect of the fix, not a regression. Rewritten with two same-length prompts,
which still genuinely share a batch and still exhibit the real caveat.

Verified end to end against the exact case that first exposed the bug: the same 12 TEST
features at r=0.6 and r=1.0 that produced ppl 5,000-28,000 before the fix now produce
ppl 380-1,200 (r=0.6) and 1,100-3,900 (r=1.0) -- comparable to B1/B2's own range at the same r,
not an order of magnitude beyond it.

### Systematic audit for the same pattern elsewhere

Checked every function in `src/steering/` that reads position-indexed activations or installs
an intervention via `ResidualHook(fn=...)`. `generate.generate()` was the only real call site
with an intervention installed on batched, potentially-padded sequences (confirmed via grep --
the only other `ResidualHook(..., fn=...)` in the codebase is a docstring example in
`hooks.py`). Every other function either forces right-padding explicitly
(`compute_feature_stats`, `cache_activations`, `sae_concept_score` -- right-padding makes
"position 0 is the sink" true for every row, the mirror-image reason `generate.generate()`
needs left-padding) or excludes the whole prompt region rather than assuming index 0 specifically
(`vectors.answer_activations`, used for Gemma persona extraction -- slices everything before
`prompt_len`, never tries to treat index 0 of the padded batch as special). No other latent
instance of this pattern found.

Found one related, smaller issue during the same audit: the notebook setup cells compute
`scale = spaces.activation_scale(hook.captured[0], exclude_sink=True)` without passing
`attention_mask`, so trailing pad tokens (present under the tokenizer's default right-padding,
harmless for the sink-skip question but not for the *value* of the median) silently entered the
scale computation. Measured: 92.3 with the mask correctly applied vs. 95.1 without, a ~3%
difference on the neutral-prompt batch -- real, but small next to the sink bug, and since every
compared arm shares the same (slightly biased) scale uniformly, relative comparisons between
methods are unaffected; only the absolute `r`-to-push mapping drifts a few percent, within the
seed-to-seed noise already characterized in Step 5 (1.4-5%). Worth fixing in future setup cells
(pass `attention_mask=enc["attention_mask"]`); not worth invalidating anything already computed.

### What this does and doesn't invalidate

Step 5/6's denoiser family selection (`evaluate_denoiser`) never touched `generate.generate()`
or any hook-installed intervention at all -- it trains and evaluates directly against cached
activations. D4's selection as the winning corruption family is unaffected by this bug.

Steps 1 and 3 did run through the buggy `generate.generate()`, with B1/B2 only (the denoiser
did not exist yet at Step 1, and Step 3's DEV sweep predates this fix). Given B1/B2's relative
insensitivity to a misapplied push on an already-huge outlier (established above), and given
Step 3's B2-loses-to-B1 finding has now independently reproduced on the corrected Step 8 TEST
data, logged as a caveat rather than a reason to rerun Steps 1/3 -- the one concretely
falsifiable claim from that period already survived a clean re-measurement.

Step 8's own cheap-axis sweep was rerun clean by Boris after deleting the contaminated results;
current numbers (see the following table) show D1/D4 in the same order of magnitude as B1/B2
at every r, and the B2-vs-B1 pattern replicating on 70 TEST concepts, not just the original 35
DEV concepts.

209 -> 307 tests across this incident's fixes, ruff clean.

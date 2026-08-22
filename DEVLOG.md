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

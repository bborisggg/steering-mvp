# Frozen decisions

Choices that must not drift, each with the reason and the date frozen. A decision recorded here
is a pre-registration: changing one after seeing a result invalidates that result and the change
must be logged in `DEVLOG.md` with what it cost.

`PLAN.md` says *what* the experiment is. This file says *exactly how*, wherever the plan leaves
room to guess.

---

## D1 — Concept usability rule (frozen before probing)

A probed SAE feature is usable if **all** of:

1. its activation frequency is inside `[0.41x, 20.5x]` of `k / d_sae`;
2. peak SAE-activation gain under an `r` sweep is `> 0.02`, restricted to `r` values that pass
   **both** `ppl_ratio <= 4x` baseline **and** `dist_2 >= 0.8x` baseline;
3. it has a Neuronpedia auto-interp description that is **not** token-level;
4. that peak reproduces under a second generation seed.

**Why in writing first:** this is the gate that defines the population every number is computed
on. Choosing it after seeing which features the method helps is the selection effect the whole
split exists to prevent.

**Why criterion 2 needs both axes, not perplexity alone (corrected 2026-08-22, before any probe
had run):** the first draft of this rule named only the perplexity bound. Reading the
exploratory repo's `evaluation/steerability.py` for the probe implementation surfaced its own
measured reason for guarding `dist_2` too: perplexity ratio and `dist_2` disagree by ~6x in
`alpha` on its 16-feature eval set, because they fail in opposite directions -- repetition
collapse drives perplexity *down* while word salad drives it *up*, so perplexity alone is
directionless and passes both failure modes silently. This project's own Step 1 gate hit exactly
this pathology (`ppl_ratio` heavy-tailed and non-monotonic across features at matched `r`, mean
13.2x vs. median 9.6x at r=1.0), which is what prompted rereading the source before implementing
the probe rather than after. `dist_2 >= 0.8x` is the exploratory repo's own calibrated
threshold, not re-derived here.

**Why these thresholds:** carried unchanged from the exploratory repo, where they produced 49
usable of 256 probed (26% raw, 19% after filters 3-4). The frequency band is expressed in
multiples of `k/d_sae` because `mean(freq) = k/d_sae` is an identity -- absolute bounds do not
transfer between SAEs, and porting them absolutely once selected the token detectors the rule
exists to exclude.

---

## D2 — What `E||h||` means in the relative strength `r`

`r = ||alpha v|| / E||h||`, where **`E||h||` is measured on the same prompt distribution being
steered**, excluding the attention sink, as a median rather than a mean.

Both the corpus value and the prompt value are recorded in the config; only the prompt value is
used.

**Why:** these differ a lot. On Gemma the exploratory repo measured 166 on the web corpus, 210 on
chat prompts, and 219 on the eval questions -- a 32% spread, which silently rescales every alpha.
Position 0 carries ~34x the norm of other tokens, so a mean is wrong by a factor of five.

---

## D3 — The conditioning signal `s`

Each corruption family has its own natural strength parameter -- `sigma` for gaussian, `beta` for
variance-preserving, `rho` for rank-1. They are **normalised to `[0, 1]` inside the family**, and
the checkpoint stores the family name plus the inverse map.

At inference the deployed `s` is derived from `r` through the stored map. A caller may not pass a
raw `s`.

**Why:** deploying a denoiser at a conditioning level it was not trained for is invisible -- the
output is plausible and wrong. In the exploratory repo the inherited alpha-to-t map turned out to
be wrong by a factor of two to three on every seed, and nobody had ever chosen it; it fell out of
notation. Storing the map with the weights makes that impossible to repeat silently.

---

## D4 — Intervention positions

**All positions**: prompt forwards and every generated-token forward -- **except position 0**
(the attention sink), which is skipped by every intervention (steering and denoising alike),
consistently.

**Why "all positions":** measured, not assumed. Restricting to generated positions removes
20-43% of the steering effect on GPT-2 and 11-31% on Gemma -- the prompt's shifted keys and
values are attended to by everything generated. Response-only is a legitimate setting but it is
a *weaker* one, and making it the default would quietly weaken every arm including the baseline.

**Why skip the sink (decided 2026-08-22):** GPT-2's position 0 carries ~34x the median token
norm (`spaces.py`, DEVLOG). Steering it alone is not inert -- 2.8-9.0% relative logit change
across r in {1, 3} -- but it is small next to steering everywhere else (24-45% at the same r's),
and the denoiser is never trained on inputs anywhere near that scale, so applying `D` there is
an out-of-distribution extrapolation with no basis for trusting the output. The exploratory repo
built this as a `skip_sink` flag on every intervention but never enabled it in any run (zero
hits in its `configs/` or `scripts/`); here it is the default, not an option, so B1/B2/D-arms
all treat position 0 the same way and no comparison is confounded by one arm touching it and
another not.

---

## D5 — Judge subset (pre-declared before any judging)

Full TEST on the cheap axis at all 10 `r`. The judge sees a **random 40 concepts of TEST** at
**4 fixed `r` values**, both drawn and fixed before the first judge call, with the seed recorded.

**Why:** judge cost is linear in concepts, and the enlarged pool would otherwise cost ~22
GPU-hours for the headline figure against ~3.6 for this subset. Pre-declaring stops it becoming a
post-hoc choice of which concepts to believe.

---

## D6 — Method freeze

At PLAN.md Step 7 the config hash is written here, with the date. After that point no TEST result
may change corruption family, pool size, architecture, training distribution, hook, `r` grid,
positions, metrics, or the judge subset.

**Frozen 2026-08-22.**

| | |
|---|---|
| Corruption | D4 — rank-1, resampled fresh every step from the full non-holdout dictionary |
| Pool | full `train_pool()` (~130,967 features) |
| Architecture | `ResidualMLPDenoiser`: d=768, hidden_mult=2, t_embed_dim=128, `center=True` |
| Training | steps=5000, batch_size=256, lr=1e-3, rho∈[0.05,3] log-uniform, winner seed=0 |
| Hook | GPT-2 small, layer 6 |
| `r` grid | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` |
| Positions | all except the attention sink |
| Metrics | reference NLL, dist-1/2/3, repetition-4, `sae_concept_score` (mechanism only), judge coherence + concept (headline) |
| Judge subset | 40 of 70 TEST concepts, seed 0, `split_fingerprint` `ce65a9496b253ceb` |
| Judge `r` values | `[0.2, 0.4, 0.6, 0.8]` |

Config hash: `49db404c4e308bc0` (`results/frozen_method_config_gpt2.json`).
Judge subset hash: `acde5e375f7dd7c0` (`results/judge_subset_gpt2.json`, concept IDs and `r`
values committed there).

Provisional choices this freeze carries forward, each already logged with its own reasoning:
D3 and D2 eliminated at Step 5 (D3's own-corruption validation loss was a memorisation artifact,
caught and corrected before it could mislead the family choice); D1-vs-D4 was a statistical tie
across two seeds, broken by the mechanistic prior that D4's corruption shape matches deployment,
not by the numbers themselves (DEVLOG 2026-08-22); Step 6 showed diversity saturates around 1024
directions, so the full pool is not doing more work than a much smaller one would, but is not
worse either.

---

## Changed after freezing

*(empty -- entries here are failures of the process, and each needs a reason)*

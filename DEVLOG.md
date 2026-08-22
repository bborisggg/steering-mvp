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
- **Python env**: the exploratory repo pins 3.12 and MPS-only (no CUDA, no bitsandbytes, no
  flash-attn). Same constraints assumed here.

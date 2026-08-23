# Handoff — 2026-08-22, end of session

## Where things stand

Step 8 (final GPT-2 test) is **in progress**. Cheap-axis sweep (B0/B1/B2/D1/D4×3seeds, all 70
TEST concepts × r-grid) is **done and clean**. Judge scoring is **paused mid-redesign** — see
"Immediate next step" below.

Steps 1–7 are complete and frozen (`DECISIONS.md` D6). Steps 9–11 (mechanism analysis, PDS
diagnostic, Gemma persona validation) **not started**.

## Read this first if anything looks confusing

**A severe bug was found and fixed today: `generate.generate()`'s sink-skip logic broke under
left-padding.** Full incident writeup: `DEVLOG.md`, search for "the sink/padding bug". One-line
version: under a mixed-length batch, position 0 is a pad token for most rows, not the true
attention sink, so the sink (30-40x normal magnitude) was getting steered/denoised like an
ordinary token — catastrophic for the denoiser (ppl >15,000), mild for B1/B2. Fixed by grouping
batches to exact tokenized length. **Confirmed isolated to this one function** (systematic
audit logged in the same DEVLOG entry) — Step 5/6's denoiser selection never went through
generation at all, so D4's selection as the winning corruption family is unaffected. Step 1/3's
old numbers were regenerated after the fix (see `git status` — several `results/*.csv` are
modified from a clean rerun, not corruption).

If you're re-deriving anything and the numbers look implausible, check whether they went
through `generate.generate()` before or after this fix (`git log` on `src/steering/generate.py`).

## Immediate next step: the judge run

Currently redesigned around `gemma4:26b` (not `gemma4:31b`) for speed — same model family,
~4-5x faster (MoE, ~4B active params), see `DEVLOG.md`/this session's chat for the benchmark
that justified it. Two-stage plan, both cached via `io.run_or_load`:

1. `results/judge_prompt_subset_gpt2.json` — the 8-of-32 prompt draw — **already exists**,
   already run.
2. `test_judge_stage1_gpt2` (8 prompts, fast check, ~1.1h projected) — **not yet saved**
   (`results/test_judge_stage1_gpt2.*` doesn't exist). Check whether it finished; if the kernel
   died mid-run, the judge's own SHA cache (`artifacts/judge_cache/`) preserves everything
   already scored, so re-running the cell picks up where it left off, not from zero.
3. `test_judge_stage2_gpt2` (remaining 24 prompts, ~3.1h) — run only after stage 1's numbers
   look sane. Then combine into `test_judge_gpt2`.

The exact three cells to paste are in this session's chat transcript (search for "For your
notebook — three cells" near the end). `notebooks/03_results.ipynb` may already have some of
this pasted in — check before re-adding.

**Before trusting the judge numbers fully**: `gemma4:26b`'s calibration against reference NLL
has not been verified — Step 3's Spearman -0.766 was measured against `gemma4:31b`. A cheap
spot-check (rerun the existing 33-example calibration sample through 26b, compare correlation)
is in the transcript, not yet run.

## Verified safe today, but check before assuming

`results/feature_splits_gpt2.json` shows as modified in `git status` — diffed it: **only the
`created` timestamp changed, dev/test/fingerprint byte-identical.** The holdout is fine. But
`vectors.save_split` (unlike `io.run_or_load`) has no overwrite protection — if Step 2's freeze
cell is ever re-run with different upstream data, it would silently overwrite the frozen split
with no error. Worth hardening at some point; not urgent since the actual holdout hasn't moved.

## Uncommitted state

`git status` has a mix of legitimately-new results (Step 8's cheap-axis sweep, concept
descriptions, the frozen method config) and refreshed old ones (Step 1/3 regenerated post-fix).
Nothing here needs to be committed *before* the next session — review and commit at a natural
stopping point, same as always (you commit, not Claude).

## Small open TODOs, not urgent

- `DECISIONS.md` D5 amendment for the 8-prompt judge subsample — write once final judge counts
  are in (flagged in DEVLOG, not yet added to DECISIONS.md).
- Notebook setup cells compute `scale` without passing `attention_mask` — ~3% bias, doesn't
  affect relative comparisons between methods, but should pass `attention_mask=enc["attention_mask"]`
  going forward. See DEVLOG for the measured numbers.
- A row-count discrepancy in the (now-discarded) revised judge cell was never resolved (5,935
  vs. an expected ≤5,440) — moot now that the cell design changed, but if it recurs, worth a
  `len()` print to chase down.

## Once the judge run finishes

The `stats.py` cells for Pareto frontier / matched-quality / matched-concept / paired bootstrap
are already written and handed off (search the transcript for "the actual primary results") —
they just need `judge_df` to exist. Run those, then move to Step 9 (mechanism analysis: pool
memorisation + what the denoiser repairs) per `PLAN.md`.

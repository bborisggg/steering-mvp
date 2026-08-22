"""Steering vectors: SAE decoder rows, persona directions, and the frozen holdout split.

**The holdout is the one property that must never be wrong.** If a corruption sampler ever
draws a DEV or TEST direction, every downstream number is invalid, silently -- the run still
produces a plausible Pareto curve. :class:`VectorSplit` makes this structural rather than only
checked: TRAIN is the *complement* of DEV/TEST over the pool, computed on demand, never stored.
A separately-maintained train list can drift out of sync with a hand-edited eval list; a
complement cannot -- there is nothing to drift.

Two independent vector sources, two independent holdout mechanisms:

- **SAE decoder rows** (GPT-2). :class:`VectorSplit` partitions feature *indices* frozen before
  any denoiser training (PLAN.md Step 2).
- **Persona directions** (Gemma). No training split is needed -- Step 11 applies the frozen
  GPT-2 recipe as-is. What persona vectors hold out instead is *questions*:
  :meth:`Trait.split_questions` keeps the evaluation half away from the extraction half, so a
  direction is judged on behaviour it was not fitted to.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import torch
from torch import nn
from tqdm.auto import tqdm

from steering import spaces
from steering.hooks import ResidualHook

# --------------------------------------------------------------------------------------------
# SAE loading
# --------------------------------------------------------------------------------------------


def load_sae(release: str, sae_id: str, layer: int, d_model: int, device: str = "cpu"):
    """Load an SAE and check it actually matches this project's model and layer convention.

    Both checks exist because a mismatch here is invisible downstream: the SAE loads, encodes,
    decodes, and every number that follows describes the wrong layer.
    """
    from sae_lens import SAE

    sae = SAE.from_pretrained(release, sae_id, device=device)
    if sae.cfg.d_in != d_model:
        raise ValueError(
            f"SAE d_in {sae.cfg.d_in} != d_model {d_model}. Wrong SAE for this model."
        )
    _assert_layer_matches(sae, layer)
    return sae


def _assert_layer_matches(sae, layer: int) -> None:
    """resid_post(L) == resid_pre(L+1) (hooks.py's convention): an SAE trained on either hook
    name is a correct match for intervention layer=L. Anything else is the off-by-one."""
    metadata = getattr(sae.cfg, "metadata", None)
    hook_name = getattr(metadata, "hook_name", None) if metadata else None
    if hook_name is None:
        return  # some releases don't record it; nothing to check
    expected = {f"blocks.{layer}.hook_resid_post", f"blocks.{layer + 1}.hook_resid_pre"}
    if hook_name not in expected:
        raise ValueError(
            f"SAE hook_name {hook_name!r} does not match intervention layer {layer}. "
            f"Expected one of {sorted(expected)}. Remember resid_post(L) == resid_pre(L+1)."
        )


# A JumpReLU SAE has no exact L0, so the check has to be a bound, on the *median* -- individual
# tokens are legitimately dense (the sink alone can fire a third of d_sae).
MAX_L0_FRACTION = 0.1


def sae_encode(sae, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """``sae.encode`` in chunks, with sparsity checked on every chunk.

    A large encode on MPS does not raise when it runs out of memory -- it returns garbage
    (measured in the exploratory repo: FVU 23.2, L0 16 under a TopK-32 SAE). Sparsity is the
    cheapest check that catches it.
    """
    k = getattr(sae.cfg, "k", None)
    out = []
    for i in range(0, x.shape[0], chunk):
        features = sae.encode(x[i : i + chunk])
        l0 = (features > 0).sum(-1)
        if k is not None and int(l0.max()) > k:
            raise RuntimeError(
                f"TopK invariant violated: max L0 {int(l0.max())} > k={k}. Likely MPS "
                f"returning corrupt output under memory pressure; lower `chunk`."
            )
        limit = sae.cfg.d_sae * MAX_L0_FRACTION
        if k is None and float(l0.float().median()) > limit:
            raise RuntimeError(
                f"SAE sparsity violated: median L0 {float(l0.float().median()):.0f} > "
                f"{limit:.0f} ({MAX_L0_FRACTION:.0%} of d_sae). Likely corrupt output; "
                f"lower `chunk`."
            )
        out.append(features)
    return torch.cat(out) if len(out) > 1 else out[0]


def steering_directions(sae, feature_ids: list[int], normalize: bool = True) -> torch.Tensor:
    """Decoder directions for the given features, as ``(n_features, d_model)``.

    Unit-normalised by default so alpha carries all the scale, comparable across features.
    """
    directions = sae.W_dec[torch.tensor(feature_ids, device=sae.W_dec.device)]
    if normalize:
        directions = directions / directions.norm(dim=-1, keepdim=True)
    return directions


# --------------------------------------------------------------------------------------------
# Feature statistics over a corpus
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def compute_feature_stats(
    model: nn.Module,
    tokenizer,
    sae,
    layer: int,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 64,
    center: bool = True,
    exclude_sink: bool = True,
) -> dict[str, torch.Tensor]:
    """Activation frequency and mean magnitude for every SAE feature, over a corpus.

    ``max_length`` should not exceed the SAE's training context: reconstruction quality falls
    off sharply past it (measured in the exploratory repo: explained variance 0.91 -> 0.31 by
    position 256-512 for a context_size=64 SAE).

    ``center`` should come from ``spaces.should_center(model_name)``, not be assumed -- see
    spaces.py. This SAE also self-normalises internally (mean-centre, divide by std, both
    undone on decode), so pre-centering composes harmlessly: a second mean-subtraction of an
    already-centered vector is a no-op, and dividing by std is invariant to a preceding
    mean-subtraction.

    Pads right regardless of the tokenizer's current ``padding_side``, restoring it afterward.
    Right-padding is what makes "position 0 is the sink" true for every row: with left-padding,
    a shorter sequence's position 0 is a pad token, not its sink, and it would silently be kept
    instead of excluded (or vice versa for a longer one). There is no shared loader in this
    project fixing `padding_side` once for every caller -- `generate.py` needs left-padding for
    its own reason -- so this function cannot trust whatever the caller left it set to.
    """
    device = next(model.parameters()).device
    fire_counts = torch.zeros(sae.cfg.d_sae, dtype=torch.float64)
    act_sums = torch.zeros(sae.cfg.d_sae, dtype=torch.float64)
    total_tokens = 0

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        starts = range(0, len(texts), batch_size)
        for start in tqdm(starts, total=len(starts), desc="feature stats", leave=False):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)

            with ResidualHook(model, layer=layer, capture=True) as hook:
                model(**enc)
            hidden = hook.captured[0]  # ResidualHook captures to CPU regardless of `device`

            start_pos = 1 if exclude_sink else 0
            # Mask is built from `enc`, which lives on `device`; `hidden` is on CPU by the
            # capture above. Move the (cheap) mask rather than the (large) hidden states.
            mask = enc["attention_mask"].bool()[:, start_pos:].to(hidden.device)
            acts = hidden[:, start_pos:, :][mask]
            if acts.numel() == 0:
                continue
            if center:
                acts = spaces.center(acts)

            # `acts` is on CPU; the SAE may not be (`sae.W_dec.device`). Move the small,
            # already-masked batch rather than requiring the SAE itself to live on CPU.
            features = sae_encode(sae, acts.to(sae.W_dec.device))
            fire_counts += (features > 0).sum(dim=0).cpu().double()
            act_sums += features.sum(dim=0).cpu().double()
            total_tokens += acts.shape[0]
    finally:
        tokenizer.padding_side = original_padding_side

    if total_tokens == 0:
        raise ValueError("no tokens processed; check the corpus, batch size and max_length")

    frequency = (fire_counts / total_tokens).float()
    mean_act = (act_sums / fire_counts.clamp_min(1)).float()
    return {
        "frequency": frequency,
        "mean_act_when_firing": mean_act,
        "n_tokens": torch.tensor(total_tokens),
        "mean_l0": torch.tensor(float(fire_counts.sum() / total_tokens)),
    }


def band_from_mean(
    stats: dict[str, torch.Tensor], lo: float, hi: float
) -> tuple[float, float]:
    """The usable frequency band in this SAE's own units.

    Mean firing frequency is an identity, not a measurement (``mean(freq) = mean_L0 / d_sae``),
    so the band transfers across SAEs only as multiples of the mean, never as absolute numbers
    -- ported absolutes selected the top 2.8% of features (token detectors) on one SAE and
    latents below the mean on another (exploratory repo, DECISIONS D1).
    """
    mean = float(stats["frequency"].mean())
    return lo * mean, hi * mean


def frequency_band(
    stats: dict[str, torch.Tensor], freq_min: float, freq_max: float
) -> torch.Tensor:
    """Feature indices whose activation frequency lies in ``[freq_min, freq_max]``.

    Below the band a feature barely fires, so any score built on it is mostly noise. Above it,
    it is typically a token-level or positional detector -- though not reliably (measured: 32%
    of in-band features are still syntactic), which is why DECISIONS D1 also requires a
    non-token-level auto-interp description.
    """
    frequency = stats["frequency"]
    return torch.nonzero((frequency >= freq_min) & (frequency <= freq_max)).squeeze(-1)


# --------------------------------------------------------------------------------------------
# VectorSplit: the frozen holdout
# --------------------------------------------------------------------------------------------


@dataclass
class VectorSplit:
    """A frozen partition of a vector pool into DEV (method selection) and TEST (final report).

    ``dev`` and ``test`` are the only sets ever stored. TRAIN is not: :meth:`train_pool` returns
    every index in ``range(pool_size)`` not in ``dev`` or ``test``, computed each time it is
    asked for. This is a deliberate departure from a stored ``train_features`` list -- storing
    it would mean either committing ~130k indices to a file that is supposed to stay small and
    diffable, or trusting a second list to stay in sync with the first. The complement can't go
    out of sync with itself.
    """

    dev: list[int]
    test: list[int]
    pool_size: int
    seed: int
    model: str
    source: str
    selection: dict = field(default_factory=dict)
    created: str = ""

    def __post_init__(self) -> None:
        dev, test = set(self.dev), set(self.test)
        overlap = dev & test
        if overlap:
            raise ValueError(
                f"HOLDOUT VIOLATION: {len(overlap)} indices appear in both dev and test: "
                f"{sorted(overlap)[:10]}"
            )
        out_of_range = {i for i in dev | test if not 0 <= i < self.pool_size}
        if out_of_range:
            raise ValueError(
                f"{len(out_of_range)} indices fall outside pool_size={self.pool_size}: "
                f"{sorted(out_of_range)[:10]}"
            )

    @property
    def holdout(self) -> set[int]:
        """Every index a corruption sampler may never draw."""
        return set(self.dev) | set(self.test)

    def train_pool(self) -> list[int]:
        return sorted(set(range(self.pool_size)) - self.holdout)

    def fingerprint(self) -> str:
        """Stable hash of dev+test, so a load can prove the frozen split was not edited."""
        payload = json.dumps({"dev": sorted(self.dev), "test": sorted(self.test)}).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "source": self.source,
            "pool_size": self.pool_size,
            "seed": self.seed,
            "created": self.created,
            "selection": self.selection,
            "n_dev": len(self.dev),
            "n_test": len(self.test),
            "dev": self.dev,
            "test": self.test,
        }


def save_split(split: VectorSplit, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split.to_dict()
    payload["fingerprint"] = split.fingerprint()
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_split(path: str | Path) -> VectorSplit:
    """Load and re-validate a frozen split -- disjointness and the fingerprint are re-checked
    here, not trusted, so a hand-edited file fails loudly rather than being used silently."""
    payload = json.loads(Path(path).read_text())
    split = VectorSplit(
        dev=payload["dev"], test=payload["test"], pool_size=payload["pool_size"],
        seed=payload["seed"], model=payload["model"], source=payload["source"],
        selection=payload.get("selection", {}), created=payload.get("created", ""),
    )
    stored = payload.get("fingerprint")
    if stored and stored != split.fingerprint():
        raise ValueError(
            f"HOLDOUT VIOLATION: split at {path} does not match its fingerprint "
            f"({split.fingerprint()} != {stored}). The frozen file was edited."
        )
    return split


# DECISIONS D1 criterion 2. RESPONSIVE_THRESHOLD and the ppl_ratio bound were the exploratory
# repo's own calibrated numbers; MIN_DIST2_RATIO too, added after this project's own Step 1
# gate independently ran into the failure it guards against (ppl_ratio heavy-tailed and
# non-monotonic across features at matched r -- DEVLOG 2026-08-22).
RESPONSIVE_THRESHOLD = 0.02
MAX_PPL_RATIO = 4.0
MIN_DIST2_RATIO = 0.8
DEFAULT_PROBE_R = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


@torch.no_grad()
def probe_steerability(
    model: nn.Module,
    tokenizer,
    sae,
    layer: int,
    feature_ids: list[int],
    prompts: list[str],
    scale: float,
    device,
    center: bool,
    r_values: tuple[float, ...] = DEFAULT_PROBE_R,
    max_new_tokens: int = 32,
    seed: int = 0,
) -> dict[int, dict]:
    """Concept response per feature over an ``r`` grid, subject to a dual fluency guard.

    Returns ``{feature_id: {peak, r_at_peak, ppl_ratio_at_peak, dist2_at_peak, baseline_act,
    baseline_fire, n_r_fluent, usable}}``, where ``peak`` is the largest SAE mean-activation
    gain over the unsteered baseline among ``r`` values that pass **both**
    ``ppl_ratio <= MAX_PPL_RATIO`` and ``dist_2 >= MIN_DIST2_RATIO * baseline_dist_2``.

    Both guards are required, not perplexity alone: they fail in opposite directions
    (repetition collapse drives perplexity down, word salad drives it up), so perplexity by
    itself passes both failure modes silently. See DECISIONS D1.

    Run *before* any denoiser exists -- this only decides which features the frozen split may
    contain, never anything about how well a method denoises them.
    """
    from steering import generate as generate_module
    from steering import metrics as metrics_module
    from steering.interventions import AdditiveSteering, NoSteering

    directions = steering_directions(sae, feature_ids)

    clean = generate_module.generate(model, tokenizer, prompts, NoSteering(), layer, device,
                                     max_new_tokens=max_new_tokens, seed=seed)
    baseline_nll = metrics_module.reference_nll(model, tokenizer, prompts, clean.texts, device)
    valid = ~baseline_nll.isnan()
    baseline_ppl = float(baseline_nll[valid].mean().exp()) if valid.any() else float("nan")
    baseline_dist2 = metrics_module.distinct_n(clean.texts, 2)

    results: dict[int, dict] = {}
    n_usable = 0
    progress = tqdm(list(zip(feature_ids, directions, strict=True)), desc="steerability")
    for feature_id, v in progress:
        base_score = metrics_module.sae_concept_score(
            model, tokenizer, sae, feature_id, layer, prompts, clean.texts, device, center
        )
        best = {
            "peak": 0.0, "r_at_peak": 0.0, "ppl_ratio_at_peak": float("nan"),
            "dist2_at_peak": float("nan"), "baseline_act": base_score["mean_act"],
            "baseline_fire": base_score["fire_rate"], "n_r_fluent": 0,
        }
        for r in r_values:
            iv = AdditiveSteering(v, r=r, scale=scale)
            out = generate_module.generate(model, tokenizer, prompts, iv, layer, device,
                                           max_new_tokens=max_new_tokens, seed=seed)
            nll = metrics_module.reference_nll(model, tokenizer, prompts, out.texts, device)
            nll_valid = ~nll.isnan()
            ppl = float(nll[nll_valid].mean().exp()) if nll_valid.any() else float("inf")
            ratio = ppl / baseline_ppl if baseline_ppl else float("inf")
            dist2 = metrics_module.distinct_n(out.texts, 2)

            if ratio > MAX_PPL_RATIO or dist2 < MIN_DIST2_RATIO * baseline_dist2:
                continue
            best["n_r_fluent"] += 1

            score = metrics_module.sae_concept_score(
                model, tokenizer, sae, feature_id, layer, prompts, out.texts, device, center
            )
            delta = score["mean_act"] - base_score["mean_act"]
            if delta > best["peak"]:
                best.update(peak=delta, r_at_peak=r, ppl_ratio_at_peak=ratio,
                           dist2_at_peak=dist2)

        best["usable"] = best["peak"] > RESPONSIVE_THRESHOLD
        results[feature_id] = best
        n_usable += int(best["usable"])
        progress.set_postfix(usable=f"{n_usable}/{len(results)}")

    return results


def select_and_freeze_split(
    stats: dict[str, torch.Tensor],
    path: str | Path,
    freq_min: float,
    freq_max: float,
    n_dev: int,
    n_test: int,
    seed: int,
    model: str,
    source: str,
    steerability: dict[int, dict] | None = None,
    descriptions: dict[int, dict] | None = None,
    replication: dict[int, float] | None = None,
    responsive_threshold: float = 0.02,
) -> VectorSplit:
    """Choose DEV and TEST, then freeze to disk. Runs once; PLAN.md Step 2 says not to revisit.

    Args:
        steerability: ``{feature_id: {"peak": float, "usable": bool, ...}}`` from an alpha-sweep
            probe (built once ``generate.py``/``interventions.py`` exist -- this function only
            consumes the result, so it has no dependency on them). When omitted, selection is
            frequency-band-only.
        descriptions: ``{feature_id: {"description": str, "token_level": bool}}``. Features
            without one, or whose description is token-level, are excluded -- the concept axis
            is judge-scored, and a judge cannot be asked about "the indefinite article 'a'".
        replication: ``{feature_id: peak_under_a_second_seed}``. Selecting the top of a large
            pool by one seed's peak inflates it (winner's curse); a feature that doesn't
            reproduce a usable response under a second seed is dropped.

    TRAIN is not chosen here at all -- it is ``pool_size`` minus whatever ends up in DEV/TEST
    (:meth:`VectorSplit.train_pool`), which is why D4's "rank-1 from the full allowed
    dictionary" can draw from far more than the frequency band this function filters by.
    """
    in_band = frequency_band(stats, freq_min, freq_max)
    pool_size = stats["frequency"].numel()

    if steerability is None:
        candidates = in_band.tolist()
        selection: dict = {
            "method": "frequency_band_only",
            "freq_min": freq_min, "freq_max": freq_max,
            "n_in_band": len(candidates),
        }
    else:
        usable = {f for f in in_band.tolist() if steerability.get(f, {}).get("usable", False)}
        n_usable_raw = len(usable)
        n_no_desc = n_token_level = n_failed_replication = 0

        if descriptions is not None:
            kept = set()
            for f in usable:
                meta = descriptions.get(f)
                if meta is None:
                    n_no_desc += 1
                elif meta["token_level"]:
                    n_token_level += 1
                else:
                    kept.add(f)
            usable = kept

        if replication is not None:
            kept = set()
            for f in usable:
                if replication.get(f, 0.0) > responsive_threshold:
                    kept.add(f)
                else:
                    n_failed_replication += 1
            usable = kept

        candidates = sorted(usable)
        selection = {
            "method": "steerability_then_description_then_replication",
            "freq_min": freq_min, "freq_max": freq_max,
            "n_in_band": int(in_band.numel()),
            "n_probed": len(steerability),
            "n_usable": n_usable_raw,
            "n_dropped_no_description": n_no_desc,
            "n_dropped_token_level": n_token_level,
            "n_dropped_failed_replication": n_failed_replication,
            "n_selectable": len(candidates),
        }
    selection["n_corpus_tokens"] = int(stats["n_tokens"].item())

    if len(candidates) < n_dev + n_test:
        raise ValueError(
            f"Only {len(candidates)} candidates survive selection; need {n_dev + n_test} "
            f"(n_dev={n_dev} + n_test={n_test}). Probe more features or widen the band."
        )

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(candidates), generator=generator).tolist()
    shuffled = [candidates[i] for i in order]
    dev = sorted(shuffled[:n_dev])
    test = sorted(shuffled[n_dev : n_dev + n_test])

    from datetime import UTC, datetime

    split = VectorSplit(
        dev=dev, test=test, pool_size=pool_size, seed=seed, model=model, source=source,
        selection=selection, created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    save_split(split, path)
    return split


# --------------------------------------------------------------------------------------------
# Persona vectors
# --------------------------------------------------------------------------------------------


@dataclass
class Trait:
    """A trait, its contrastive prompt pair, and the questions used to elicit it."""

    name: str
    description: str
    positive: list[str]
    negative: list[str]
    questions: list[str]

    def split_questions(self, n_extract: int) -> tuple[list[str], list[str]]:
        """Extraction half and evaluation half. The direction never sees the eval questions."""
        return self.questions[:n_extract], self.questions[n_extract:]


def load_traits(path: str | Path, names: list[str] | None = None) -> dict[str, Trait]:
    payload = json.loads(Path(path).read_text())
    names = names or list(payload)
    return {n: Trait(name=n, **payload[n]) for n in names}


def has_chat_template(tokenizer) -> bool:
    """Whether this tokenizer formats conversations at all.

    Distinct from :func:`supports_system_role`: Gemma has a template but no system turn, GPT-2
    has neither.
    """
    try:
        tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False)
        return True
    except Exception:  # noqa: BLE001 -- template failure modes vary by tokenizer backend
        return False


def supports_system_role(tokenizer) -> bool:
    """Whether the chat template accepts a system turn. Gemma's raises; checked, not assumed."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tokenize=False,
        )
        return True
    except Exception:  # noqa: BLE001 -- template failure modes vary by tokenizer backend
        return False


def build_prompt(tokenizer, instruction: str, question: str) -> str:
    """Format one (instruction, question) pair for this model's chat template."""
    if instruction and supports_system_role(tokenizer):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ]
    elif instruction:
        messages = [{"role": "user", "content": f"{instruction}\n\n{question}"}]
    else:
        messages = [{"role": "user", "content": question}]
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[-1]["content"]


def _eos_ids(tokenizer) -> set[int]:
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for token in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid >= 0:
            ids.add(tid)
    return {i for i in ids if i is not None}


def _answer_mask(generated: torch.Tensor, prompt_len: int, eos_ids: set[int]) -> torch.Tensor:
    """True on real answer tokens: generated, and before the first end-of-turn."""
    answer = generated[:, prompt_len:]
    is_eos = torch.zeros_like(answer, dtype=torch.bool)
    for tid in eos_ids:
        is_eos |= answer == tid
    return is_eos.cumsum(dim=1) == 0  # everything from the first EOS on is unchosen padding


@torch.no_grad()
def answer_activations(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    layer: int,
    device,
    center: bool,
    max_new_tokens: int = 48,
    batch_size: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    """Per-prompt mean residual activation over the answer, plus the answers themselves.

    Averaged over what the model *writes*, not the prompt: the trait lives in the response.
    Prompt positions and post-EOS padding are excluded. ``center`` must come from
    ``spaces.should_center(model_name)`` -- see spaces.py; centering a Gemma-family model here
    would perturb the very activations being measured.
    """
    means, texts = [], []
    eos = _eos_ids(tokenizer)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        prompt_len = enc["input_ids"].shape[1]
        torch.manual_seed(seed + start)
        generated = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)

        attention = (generated != tokenizer.pad_token_id).long()
        attention[:, :prompt_len] = enc["attention_mask"]
        with ResidualHook(model, layer=layer, capture=True) as hook:
            model(generated, attention_mask=attention)
        # ResidualHook captures to CPU regardless of `device`; `generated` (and anything built
        # from it) lives on `device`, so the mask has to follow the hidden states, not the
        # other way round.
        hidden = hook.captured[0][:, prompt_len:, :].float()
        if center:
            hidden = spaces.center(hidden)

        mask = _answer_mask(generated, prompt_len, eos).unsqueeze(-1).to(hidden.device)
        totals = (hidden * mask).sum(1)
        counts = mask.sum(1).clamp_min(1)
        means.append((totals / counts).cpu())
        texts += tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
    return torch.cat(means), texts


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Standardised mean difference (persona-vectors paper's separation measure)."""
    pooled = ((a.var(unbiased=True) + b.var(unbiased=True)) / 2).sqrt().clamp_min(1e-8)
    return float((a.mean() - b.mean()) / pooled)


def extract_persona_vector(
    model: nn.Module,
    tokenizer,
    trait: Trait,
    layer: int,
    device,
    center: bool = True,
    n_extract: int = 20,
    max_new_tokens: int = 48,
    seed: int = 0,
) -> dict:
    """Difference in mean answer activation between eliciting and suppressing prompts.

    Reports separation on the questions it was fitted to; use :func:`separation` for the
    held-out half.
    """
    extract_qs, _ = trait.split_questions(n_extract)
    pos_prompts = [build_prompt(tokenizer, i, q) for i in trait.positive for q in extract_qs]
    neg_prompts = [build_prompt(tokenizer, i, q) for i in trait.negative for q in extract_qs]

    pos, _ = answer_activations(model, tokenizer, pos_prompts, layer, device, center,
                                max_new_tokens, seed=seed)
    neg, _ = answer_activations(model, tokenizer, neg_prompts, layer, device, center,
                                max_new_tokens, seed=seed)
    direction = pos.mean(0) - neg.mean(0)
    unit = direction / direction.norm()
    return {
        "trait": trait.name,
        "description": trait.description,
        "layer": layer,
        "vector": unit.tolist(),
        "raw_norm": float(direction.norm()),
        "train_cohens_d": cohens_d(pos @ unit, neg @ unit),
        "n_extract_questions": len(extract_qs),
        "n_positive_prompts": len(pos_prompts),
        "n_negative_prompts": len(neg_prompts),
    }


def separation(
    model: nn.Module,
    tokenizer,
    trait: Trait,
    direction: torch.Tensor,
    layer: int,
    device,
    center: bool = True,
    n_extract: int = 20,
    max_new_tokens: int = 48,
    seed: int = 0,
) -> dict:
    """How well the direction separates trait-on from trait-off answers it never saw.

    A direction that only separates its own extraction questions has fitted the questions, not
    the trait.
    """
    _, eval_qs = trait.split_questions(n_extract)
    unit = (direction / direction.norm()).cpu()
    pos_prompts = [build_prompt(tokenizer, i, q) for i in trait.positive for q in eval_qs]
    neg_prompts = [build_prompt(tokenizer, i, q) for i in trait.negative for q in eval_qs]

    pos, _ = answer_activations(model, tokenizer, pos_prompts, layer, device, center,
                                max_new_tokens, seed=seed)
    neg, _ = answer_activations(model, tokenizer, neg_prompts, layer, device, center,
                                max_new_tokens, seed=seed)
    p, n = pos @ unit, neg @ unit
    return {
        "trait": trait.name,
        "layer": layer,
        "n_eval_questions": len(eval_qs),
        "cohens_d": cohens_d(p, n),
        "auc": float((p[:, None] > n[None, :]).float().mean()),
        "mean_projection_positive": float(p.mean()),
        "mean_projection_negative": float(n.mean()),
    }


def save_persona_vectors(vectors: dict[str, dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vectors, indent=2) + "\n")


def load_persona_vectors(path: str | Path) -> dict[str, torch.Tensor]:
    payload = json.loads(Path(path).read_text())
    return {name: torch.tensor(entry["vector"]) for name, entry in payload.items()}


# --------------------------------------------------------------------------------------------
# Neuronpedia: auto-interp descriptions, for DECISIONS D1 criterion 3
# --------------------------------------------------------------------------------------------
# An SAE feature has an index, not a name, so there is nothing to ask a judge about.
# Neuronpedia hosts auto-interp descriptions for the SAE used here, which is what makes a
# judge-scored concept axis possible on GPT-2 at all. The description says what makes a
# feature *fire*, not what steering with it *produces* -- a deliberately imperfect proxy.
#
# Cached to disk, one file per feature: responses never change and the API is rate-limited.
# ``requests``, not ``urllib`` -- macOS Python has no CA bundle and urllib fails with
# CERTIFICATE_VERIFY_FAILED, while requests bundles certifi.

NEURONPEDIA_API = "https://www.neuronpedia.org/api/feature"

# Descriptions naming a token/syntax pattern rather than a semantic concept -- a judge cannot
# be asked whether text "expresses the indefinite article 'a'". Ported from the exploratory
# repo including its bug fix: the trailing ``s?`` is load-bearing. An earlier version ended
# each alternative at a word boundary and missed every plural; "articles and indefinite
# articles" slipped through and was selected into a frozen split at the third-highest peak.
_TOKEN_LEVEL = re.compile(
    r"\b(?:"
    r"punctuation|comma|semicolon|colon|period|apostrophe|quotation|bracket|parenthes\w*|"
    r"whitespace|newline|line break|"
    r"(?:in)?definite article|the article|"
    r"the word ['\"]|the term ['\"]|the phrase ['\"]|"
    r"conjunction|preposition|pronoun|determiner|"
    r"suffix|prefix|token|morpheme|"
    r"capital letter|letter ['\"]|single character|special character|non-english character"
    r")s?\b",
    re.IGNORECASE,
)


def is_token_level(description: str) -> bool:
    """Whether a description names a token/syntax pattern rather than a semantic concept."""
    return bool(_TOKEN_LEVEL.search(description))


def fetch_feature(
    neuronpedia_id: str,
    index: int,
    cache_dir: str | Path,
    timeout: int = 25,
    pause: float = 0.4,
) -> dict | None:
    """One feature's Neuronpedia record, cached. ``None`` if it cannot be fetched.

    ``cache_dir`` is an explicit argument, not defaulted to ``io.ARTIFACTS`` -- consistent with
    every other function in this project that reads external state, the caller says where.
    """
    model_id, source = neuronpedia_id.split("/", 1)
    path = Path(cache_dir) / source / f"{index}.json"
    if path.exists():
        return json.loads(path.read_text())

    try:
        r = requests.get(f"{NEURONPEDIA_API}/{model_id}/{source}/{index}", timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception:  # noqa: BLE001 -- any failure here means "skip this feature", not raise
        return None
    time.sleep(pause)  # be polite to a free service

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def describe(payload: dict | None) -> str | None:
    """The auto-interp description from a feature record, if it has one."""
    if not payload:
        return None
    explanations = payload.get("explanations") or []
    if not explanations:
        return None
    description = (explanations[0].get("description") or "").strip()
    return description or None


def describe_features(
    neuronpedia_id: str, indices: list[int], cache_dir: str | Path
) -> dict[int, dict]:
    """Descriptions for many features: ``{index: {description, token_level, frac_nonzero}}``.

    Features with no description are omitted -- they cannot be judged, so they cannot become
    DEV or TEST features (DECISIONS D1 criterion 3).
    """
    out = {}
    for index in tqdm(indices, desc="neuronpedia", leave=False):
        payload = fetch_feature(neuronpedia_id, index, cache_dir)
        description = describe(payload)
        if description is None:
            continue
        out[index] = {
            "description": description,
            "token_level": is_token_level(description),
            "frac_nonzero": payload.get("frac_nonzero"),
        }
    return out

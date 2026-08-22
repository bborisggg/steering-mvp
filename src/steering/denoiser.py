"""The residual MLP: D(x, s) = x + f(N(x), e(s)), output head zero-initialised.

Zero-init means an untrained denoiser is exactly the identity, which is the correct prior and a
test.

``cache_activations`` (PLAN.md Step 4) is the training data this module's architecture will
consume in Step 5. It stores **raw** activations, not pre-centered or pre-scaled ones: the
encode step belongs inside ``D()``'s own forward pass (that is what ``N(x)`` is, via
``spaces.encode``), so it must run identically at training and at inference through the hook.
Baking it into the cache instead would mean the transform ran once, at cache time, and never
again -- exactly the opposite of what "training and inference share the same transform" (the
step's own requirement) is protecting against.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from tqdm.auto import tqdm

from steering.hooks import ResidualHook


@torch.no_grad()
def cache_activations(
    model: nn.Module,
    tokenizer,
    layer: int,
    texts: list[str],
    max_tokens: int,
    batch_size: int = 16,
    max_length: int = 64,
    exclude_sink: bool = True,
) -> torch.Tensor:
    """Raw residual-stream activations at ``layer``, as ``[N, d_model]`` with ``N <= max_tokens``.

    Stops once ``max_tokens`` is reached rather than scanning the whole corpus -- this builds a
    training *pool*, not a corpus census (contrast ``vectors.compute_feature_stats``, which
    always scans everything given for an accurate frequency estimate). ``exclude_sink=True`` by
    default: every intervention skips the sink (DECISIONS D4), so the denoiser should never be
    trained to reconstruct a position it will never be asked to touch at inference.

    Pads right regardless of the caller's current ``padding_side``, restoring it afterward --
    same reason as ``compute_feature_stats``: right-padding is what makes "position 0 is the
    sink" true for every row.
    """
    device = next(model.parameters()).device
    chunks: list[torch.Tensor] = []
    n_collected = 0

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        starts = range(0, len(texts), batch_size)
        for start in tqdm(starts, total=len(starts), desc="caching activations", leave=False):
            if n_collected >= max_tokens:
                break
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)

            with ResidualHook(model, layer=layer, capture=True) as hook:
                model(**enc)
            hidden = hook.captured[0]  # ResidualHook captures to CPU regardless of `device`

            start_pos = 1 if exclude_sink else 0
            # Mask is built from `enc`, which lives on `device`; move it to `hidden`'s device
            # rather than the (larger) hidden states -- same fix as compute_feature_stats.
            mask = enc["attention_mask"].bool()[:, start_pos:].to(hidden.device)
            acts = hidden[:, start_pos:, :][mask]
            if acts.numel() == 0:
                continue

            remaining = max_tokens - n_collected
            if acts.shape[0] > remaining:
                acts = acts[:remaining]
            chunks.append(acts)
            n_collected += acts.shape[0]
    finally:
        tokenizer.padding_side = original_padding_side

    if not chunks:
        raise ValueError("no activations collected; check the corpus, batch size and max_length")
    return torch.cat(chunks)


# ==============================================================================================
# ResidualMLPDenoiser: D(x, s) = x + f(N(x), e(s))    (PLAN.md Section 3)
# ==============================================================================================
# "normalize" in PLAN.md's per-layer spec is spaces.encode -- applied once, at the block's
# input, mapping raw x into the O(1)-scaled space the MLP body operates in throughout. There is
# no second, internal LayerNorm: PLAN.md's list names "normalize" exactly once, and adding a
# further normalisation step inside the block is machinery the spec does not call for.
#
# The MLP body's output is a correction in *normalised* units. Adding it directly to raw `x`
# would be the un-scale trap named in this project's own conventions -- "forgetting the
# un-scale produces plausible-looking but wrong Pareto curves," not a crash. So the correction
# is passed back through the scale multiply (spaces.decode's own operation) before the residual
# add, making training and inference share literally the same spaces.py calls rather than two
# separately-written implementations of the same arithmetic.
#
# Conditioning mechanism (FiLM over a Fourier embedding of `t`) is not specified by PLAN.md,
# which only requires "condition on corruption strength" -- ported from the exploratory repo,
# a proven, standard choice for this exact purpose (Boris: "consult previous repos where
# needed"), inserted once, after the block's first activation.

from steering import spaces


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal embedding of a scalar level in [0,1], as used by diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        return torch.cat([args.cos(), args.sin()], dim=-1).to(t.dtype)


class ResidualMLPDenoiser(nn.Module):
    """``D(x, s) = x + scale * f(encode(x, scale), e(s))``, one residual block, output head
    zero-initialised so an untrained denoiser is exactly the identity.

    Args:
        d_model: residual-stream width.
        activation_scale: ``spaces.activation_scale`` at the intervention layer, on the same
            prompt distribution being steered (DECISIONS D2). Stored in the checkpoint so
            inference cannot silently use a different one.
        center: whether this architecture's residual stream may be centered
            (``spaces.should_center(model_name)`` -- GPT-2 True, Gemma False). Stored, not
            re-derived, so a checkpoint is self-describing.
        hidden_mult: PLAN.md's spec is ``Linear(d->2d)...Linear(2d->d)``, i.e. ``hidden_mult=2``;
            exposed as a parameter rather than hardcoded so the architecture stays inspectable.
        t_embed_dim: width of the Fourier time embedding feeding the FiLM modulation.
    """

    def __init__(
        self,
        d_model: int,
        activation_scale: float,
        center: bool,
        hidden_mult: int = 2,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.center = center
        self.register_buffer("activation_scale", torch.tensor(float(activation_scale)))

        hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.act1 = nn.SiLU()
        self.t_embed = FourierTimeEmbedding(t_embed_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim), nn.SiLU(),
            nn.Linear(t_embed_dim, 2 * hidden),  # FiLM: one scale, one shift, per hidden unit
        )
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.SiLU()
        self.out = nn.Linear(hidden, d_model)

        # Zero-init the head: f = 0 at step 0, so D(x, s) = x + scale*0 = x. Training starts
        # from the identity, the correct prior for a denoiser and the module's own test.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def config(self) -> dict[str, object]:
        return {
            "d_model": self.d_model,
            "activation_scale": float(self.activation_scale),
            "center": self.center,
            "hidden_mult": self.fc1.out_features // self.d_model,
            "t_embed_dim": self.t_embed.dim,
        }

    def correction(self, encoded: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``f(N(x), e(s))``, entirely in normalised units -- the caller un-scales."""
        t_emb = self.t_mlp(self.t_embed(t))
        while t_emb.dim() < encoded.dim():
            t_emb = t_emb.unsqueeze(1)
        film_scale, film_shift = t_emb.chunk(2, dim=-1)

        h = self.act1(self.fc1(encoded))
        h = h * (1.0 + film_scale) + film_shift
        h = self.act2(self.fc2(h))
        return self.out(h)

    def forward(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        """Denoise. ``x`` is a raw activation in model units, any leading batch shape."""
        if isinstance(t, (int, float)):
            t = torch.full(x.shape[:1], float(t), device=x.device, dtype=x.dtype)

        scale = float(self.activation_scale)
        encoded = spaces.encode(x, scale, center=self.center)
        correction = self.correction(encoded, t)
        # The un-scale: correction is in normalised units, x is raw. Skipping this line is
        # exactly the trap named in the module docstring above. spaces.decode(v, scale) is
        # exactly `v * scale` (its `center` argument is a no-op there, by spaces.py's own
        # design -- centering is never undone on decode); used here anyway so this is
        # literally the same call site the rest of the project uses, not a second,
        # separately-written copy of the same arithmetic.
        return x + spaces.decode(correction, scale, center=self.center)


def denoising_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``L = ||pred - target||^2 / (||target||^2 + eps)`` (PLAN.md Section 3), mean over batch.

    Normalised by the target's own squared norm so tokens of different natural scale
    contribute comparably -- an un-normalised MSE would be dominated by whichever tokens
    happen to have the largest activations.
    """
    diff_sq = (pred - target).pow(2).sum(dim=-1)
    target_sq = target.pow(2).sum(dim=-1)
    return (diff_sq / (target_sq + eps)).mean()


def save_denoiser(model: ResidualMLPDenoiser, path, extra: dict | None = None) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.config, "state_dict": model.state_dict(), "extra": extra or {}}, path
    )


def load_denoiser(path, device: str | None = None) -> ResidualMLPDenoiser:
    """Rebuild a checkpoint's architecture from its own stored config -- never guessed."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    cfg = payload["config"]
    model = ResidualMLPDenoiser(
        d_model=cfg["d_model"], activation_scale=cfg["activation_scale"], center=cfg["center"],
        hidden_mult=cfg["hidden_mult"], t_embed_dim=cfg["t_embed_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.requires_grad_(False)
    if device:
        model.to(device)
    return model


def train_denoiser(
    activations: torch.Tensor,
    corruption,
    d_model: int,
    activation_scale: float,
    center: bool,
    steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log_every: int = 200,
) -> tuple[ResidualMLPDenoiser, list[dict]]:
    """Train one denoiser against one corruption family (PLAN.md Step 5).

    ``activations`` is the raw ``[N, d_model]`` pool from ``cache_activations`` -- always on
    CPU, regardless of ``device`` (``ResidualHook`` captures to CPU by design). Only the sampled
    minibatch is moved to ``device`` each step, not the whole pool.

    Index sampling uses a CPU generator; the corruption's own noise uses a separate generator on
    ``device``, since ``corruptions._check_generator`` requires it to match the tensor it
    operates on (the minibatch, once moved). Same seed value for both, different generator
    objects -- no collision, since they draw independent streams for independent purposes.

    Returns the trained model (``eval()``, gradients disabled -- matching ``load_denoiser``'s
    convention for anything handed back for evaluation) and a training history: loss on the
    minibatch, plus ``identity_gap`` -- ``||D(h,0)-h||`` on a genuinely clean sample at the
    least-corrupted conditioning level (PLAN.md: "monitor ... add no extra regularizer unless
    clean states are measurably damaged" -- this is what that monitoring is).
    """
    torch.manual_seed(seed)
    model = ResidualMLPDenoiser(d_model, activation_scale, center).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    idx_generator = torch.Generator().manual_seed(seed)
    corruption_generator = torch.Generator(device=device).manual_seed(seed)

    n = activations.shape[0]
    history: list[dict] = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=idx_generator)
        clean = activations[idx].to(device)
        corrupted = corruption(clean, generator=corruption_generator)

        pred = model(corrupted.x, corrupted.t)
        loss = denoising_loss(pred, corrupted.target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                identity_gap = float((model(clean, t=0.0) - clean).norm(dim=-1).mean())
            history.append({"step": step, "loss": float(loss.detach()),
                           "identity_gap": identity_gap})

    model.eval()
    model.requires_grad_(False)
    return model, history


@torch.no_grad()
def evaluate_denoiser(
    model: ResidualMLPDenoiser,
    activations: torch.Tensor,
    corruption,
    n_examples: int = 4096,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """Mean ``denoising_loss`` over ``n_examples`` freshly corrupted from ``activations``.

    Pass a validation ``seed`` distinct from whatever seed trained ``model`` -- evaluating on
    the exact same (index, noise) draws used during training would reward memorising those
    particular corruption instances, not the underlying denoising task, and this is exactly the
    number PLAN.md Step 5's "eliminate clearly weak families" decision reads.
    """
    model.eval()
    idx_generator = torch.Generator().manual_seed(seed)
    corruption_generator = torch.Generator(device=device).manual_seed(seed)
    n = activations.shape[0]

    total_loss, total_n, seen = 0.0, 0, 0
    while seen < n_examples:
        this_batch = min(batch_size, n_examples - seen)
        idx = torch.randint(0, n, (this_batch,), generator=idx_generator)
        clean = activations[idx].to(device)
        corrupted = corruption(clean, generator=corruption_generator)
        pred = model(corrupted.x, corrupted.t)
        loss = denoising_loss(pred, corrupted.target)
        total_loss += float(loss) * this_batch
        total_n += this_batch
        seen += this_batch
    return total_loss / total_n

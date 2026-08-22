"""Reading and rewriting the residual stream at one block.

Plain ``register_forward_hook``. No TransformerLens: it is slow and flaky on MPS, and the only
thing needed here is one tensor at one place.

**Layer convention.** ``layer=L`` means the *output* of block ``L``. Equivalently
``resid_post(L)``, ``resid_pre(L+1)``, and ``hidden_states[L+1]`` from a forward pass with
``output_hidden_states=True`` -- the offset because ``hidden_states[0]`` is the embedding output.
The equivalence stops at the last block, whose entry in that tuple has the final norm applied
(GPT-2's ``ln_f``). This project always intervenes mid-depth.

**Generation.** With a KV cache the first forward carries the whole prompt and each later forward
carries a single token. A hook that fires only on the first pass still changes the generated text,
because the prompt's activations sit in the cache that every later token attends to. So "the
output changed" is not evidence the intervention reached generation. :attr:`ResidualHook.seq_lens`
records the sequence length of every pass so that can be asserted instead of assumed.

**What ``fn`` may return.** The same shape, dtype and device, or ``None`` for no change. A dtype
change is rejected rather than cast: the denoiser runs in fp32 and a bf16 model must be converted
somewhere explicit, which is ``spaces.py``, not silently here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import torch
from torch import nn

# Where the decoder blocks live, by architecture family. GPT-2 and Gemma are the two this
# project uses; the rest cost nothing and save a debugging session if the scope ever moves.
BLOCK_PATHS = (
    "transformer.h",         # GPT-2, GPT-J
    "model.layers",          # Gemma, Llama, Mistral, Qwen
    "gpt_neox.layers",       # NeoX, Pythia
    "model.decoder.layers",  # OPT
)


def get_blocks(model: nn.Module) -> nn.ModuleList:
    """The ModuleList of decoder blocks, whatever this architecture calls it."""
    for path in BLOCK_PATHS:
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList):
            return node
    raise AttributeError(
        f"No decoder blocks found on {type(model).__name__}. Tried {', '.join(BLOCK_PATHS)}. "
        f"Add this architecture to BLOCK_PATHS."
    )


def n_layers(model: nn.Module) -> int:
    return len(get_blocks(model))


def hidden_state_index(layer: int) -> int:
    """Index into HF's ``hidden_states`` tuple holding the output of block ``layer``."""
    return layer + 1


def _hidden_of(output: Any) -> torch.Tensor:
    """The hidden state inside a block's output, which may be a bare tensor or a tuple."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Cannot find a hidden state in block output of type {type(output)!r}")


def _with_hidden(output: Any, hidden: torch.Tensor) -> Any:
    """Rebuild a block's output around a new hidden state, preserving the container type."""
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return [hidden, *output[1:]]


class ResidualHook:
    """Read and/or rewrite the residual stream at the output of one block.

    Context manager, always::

        with ResidualHook(model, layer=6, fn=lambda h: h + alpha * v) as hook:
            out = model.generate(**batch, max_new_tokens=32)
        assert hook.n_decode_steps > 0  # the intervention reached generation

    Args:
        model: the causal LM.
        layer: block index. The hook sees that block's output; see the module docstring.
        fn: maps the hidden state to its replacement, or returns ``None`` to leave it alone.
        capture: store every hidden state the block produces, *before* ``fn`` is applied. To
            record what was substituted instead, capture from inside ``fn``.
        capture_device: where captures are kept. CPU by default -- caching a corpus of
            activations on an MPS device runs it out of memory well before the corpus is useful.

    Attributes:
        n_calls: number of forward passes seen.
        seq_lens: sequence length of each pass, in order. During cached generation this is
            ``[prompt_len, 1, 1, ...]``, which is what proves the hook reached the decode passes.
        captured: the hidden states, if ``capture``.
    """

    def __init__(
        self,
        model: nn.Module,
        layer: int,
        fn: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
        capture: bool = False,
        capture_device: str | torch.device = "cpu",
    ) -> None:
        blocks = get_blocks(model)
        if not -len(blocks) <= layer < len(blocks):
            raise IndexError(f"layer {layer} out of range for {len(blocks)} blocks")
        self.layer = layer
        self.fn = fn
        self.capture = capture
        self.capture_device = capture_device
        self.captured: list[torch.Tensor] = []
        self.seq_lens: list[int] = []
        self.n_calls = 0
        self._block = blocks[layer]
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    # --- read-only views over the call trace ---

    @property
    def prefill_len(self) -> int | None:
        """Sequence length of the first pass: the prompt, under a KV cache."""
        return self.seq_lens[0] if self.seq_lens else None

    @property
    def n_decode_steps(self) -> int:
        """Passes carrying a single token -- one per generated token after the first."""
        return sum(1 for n in self.seq_lens[1:] if n == 1)

    # --- the hook itself ---

    def _run(self, module: nn.Module, args: tuple, output: Any) -> Any:
        hidden = _hidden_of(output)
        self.n_calls += 1
        self.seq_lens.append(hidden.shape[1])
        if self.capture:
            self.captured.append(hidden.detach().to(self.capture_device).clone())
        if self.fn is None:
            return output

        replacement = self.fn(hidden)
        if replacement is None:
            return output
        if replacement.shape != hidden.shape:
            raise ValueError(
                f"fn changed the shape at layer {self.layer}: "
                f"{tuple(hidden.shape)} -> {tuple(replacement.shape)}"
            )
        if replacement.dtype != hidden.dtype:
            raise ValueError(
                f"fn changed the dtype at layer {self.layer}: {hidden.dtype} -> "
                f"{replacement.dtype}. Cast deliberately in spaces.py, not here."
            )
        if replacement.device != hidden.device:
            raise ValueError(
                f"fn changed the device at layer {self.layer}: {hidden.device} -> "
                f"{replacement.device}"
            )
        return _with_hidden(output, replacement)

    def reset(self) -> None:
        """Clear the call trace and captures, leaving the hook registered."""
        self.n_calls = 0
        self.seq_lens.clear()
        self.captured.clear()

    def __enter__(self) -> Self:
        self._handle = self._block.register_forward_hook(self._run)
        return self

    def __exit__(self, *exc: object) -> None:
        # Unconditional: a hook leaked by an exception silently steers every later forward in
        # the notebook session, and nothing about the symptom points back here.
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

"""The hook is where off-by-one errors become invisible.

Two failures this pins, both of which produce plausible numbers rather than crashes:

1. **Layer off-by-one.** ``resid_post`` of block 6 is ``resid_pre`` of block 7 is
   ``hidden_states[7]``. Steering one block away from where the vector was extracted still
   moves the output, just less well, so the Pareto curve looks merely disappointing.
2. **Prompt-only intervention.** Under a KV cache the prefill pass carries the whole prompt and
   every later pass carries one token. A hook that only reaches prefill still changes the
   generation -- through the cache -- so the text does change and nothing looks wrong.

Run against real GPT-2, not a stub: the point is the convention of this specific architecture.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from steering import hooks

LAYER = 6  # mid-depth on GPT-2 small; the frozen hook point for this project


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    return model, tok


@pytest.fixture(scope="module")
def batch(gpt2):
    _, tok = gpt2
    return tok(["The capital of France is"], return_tensors="pt")


# --- locating the blocks -------------------------------------------------------------------

def test_finds_gpt2_blocks(gpt2):
    model, _ = gpt2
    blocks = hooks.get_blocks(model)
    assert isinstance(blocks, nn.ModuleList)
    assert len(blocks) == 12 == hooks.n_layers(model)


def test_unknown_architecture_raises_with_the_paths_it_tried():
    with pytest.raises(AttributeError, match="transformer.h"):
        hooks.get_blocks(nn.Linear(4, 4))


@pytest.mark.parametrize("layer", [12, -13, 99])
def test_layer_out_of_range_raises(gpt2, layer):
    model, _ = gpt2
    with pytest.raises(IndexError, match="out of range"):
        hooks.ResidualHook(model, layer=layer)


# --- the layer convention ------------------------------------------------------------------

def test_capture_at_layer_L_equals_hidden_state_L_plus_1(gpt2, batch):
    """layer=L means the *output* of block L. THE off-by-one test."""
    model, _ = gpt2
    with torch.no_grad():
        reference = model(**batch, output_hidden_states=True).hidden_states

    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook, torch.no_grad():
        model(**batch)

    assert len(hook.captured) == 1
    torch.testing.assert_close(hook.captured[0], reference[LAYER + 1])
    assert hooks.hidden_state_index(LAYER) == LAYER + 1


def test_capture_at_layer_L_is_not_hidden_state_L(gpt2, batch):
    """The neighbouring index must be visibly different, or the test above proves nothing."""
    model, _ = gpt2
    with torch.no_grad():
        reference = model(**batch, output_hidden_states=True).hidden_states
    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook, torch.no_grad():
        model(**batch)
    assert not torch.allclose(hook.captured[0], reference[LAYER], atol=1e-3)


def test_last_block_is_not_the_final_hidden_state(gpt2, batch):
    """``hidden_states[-1]`` has ``ln_f`` applied, so the equivalence stops at the last block.

    Documented rather than worked around: this project always intervenes mid-depth.
    """
    model, _ = gpt2
    last = hooks.n_layers(model) - 1
    with torch.no_grad():
        reference = model(**batch, output_hidden_states=True).hidden_states
    with hooks.ResidualHook(model, layer=last, capture=True) as hook, torch.no_grad():
        model(**batch)
    assert not torch.allclose(hook.captured[0], reference[-1], atol=1e-3)


# --- firing throughout generation ------------------------------------------------------------

def test_fires_once_per_forward_with_the_expected_shapes(gpt2, batch):
    """One prefill pass carrying the prompt, then one pass per new token."""
    model, _ = gpt2
    prompt_len = batch["input_ids"].shape[1]
    n_new = 8

    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook:
        model.generate(**batch, max_new_tokens=n_new, do_sample=False, use_cache=True,
                       pad_token_id=50256)

    assert hook.seq_lens == [prompt_len] + [1] * (n_new - 1)
    assert hook.n_calls == n_new
    assert hook.prefill_len == prompt_len
    assert hook.n_decode_steps == n_new - 1


def test_intervention_reaches_generated_positions_not_only_the_prompt(gpt2, batch):
    """The trap: a prompt-only hook still changes the output, through the KV cache.

    So "the text changed" does not prove the intervention reached generation. Only comparing
    against a deliberately prompt-only hook does.
    """
    model, _ = gpt2
    torch.manual_seed(0)
    direction = torch.randn(768) * 0.5

    def everywhere(hidden):
        return hidden + direction

    def prefill_only(hidden):
        return hidden + direction if hidden.shape[1] > 1 else None

    outputs = {}
    for name, fn in (("all", everywhere), ("prefill", prefill_only)):
        with hooks.ResidualHook(model, layer=LAYER, fn=fn):
            outputs[name] = model.generate(**batch, max_new_tokens=12, do_sample=False,
                                           use_cache=True, pad_token_id=50256)

    assert not torch.equal(outputs["all"], outputs["prefill"]), (
        "steering all positions matched steering only the prompt -- the hook is not reaching "
        "the decode passes"
    )


def test_fn_returning_none_is_exactly_a_no_op(gpt2, batch):
    model, _ = gpt2
    with torch.no_grad():
        clean = model(**batch).logits
    with hooks.ResidualHook(model, layer=LAYER, fn=lambda h: None), torch.no_grad():
        hooked = model(**batch).logits
    torch.testing.assert_close(clean, hooked)


def test_fn_changes_the_logits(gpt2, batch):
    model, _ = gpt2
    with torch.no_grad():
        clean = model(**batch).logits
    with hooks.ResidualHook(model, layer=LAYER, fn=lambda h: h * 1.5), torch.no_grad():
        steered = model(**batch).logits
    assert not torch.allclose(clean, steered, atol=1e-3)


# --- guards on what fn returns ---------------------------------------------------------------

def test_shape_change_raises(gpt2, batch):
    """A broadcasting slip would otherwise propagate into the rest of the network."""
    model, _ = gpt2
    with pytest.raises(ValueError, match="shape"), \
            hooks.ResidualHook(model, layer=LAYER, fn=lambda h: h[:, :1, :]):
        model(**batch)


def test_dtype_change_raises(gpt2, batch):
    """The denoiser runs in fp32; a bf16 model must be converted deliberately, in spaces.py."""
    model, _ = gpt2
    with pytest.raises(ValueError, match="dtype"), \
            hooks.ResidualHook(model, layer=LAYER, fn=lambda h: h.to(torch.float64)):
        model(**batch)


# --- capture and lifecycle ---------------------------------------------------------------

def test_capture_is_detached_and_on_cpu(gpt2, batch):
    model, _ = gpt2
    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook:
        model(**batch)
    captured = hook.captured[0]
    assert captured.device.type == "cpu"
    assert not captured.requires_grad
    assert captured.grad_fn is None


def test_capture_records_the_stream_before_fn(gpt2, batch):
    """``captured`` is what the block produced, not what was substituted for it.

    Compared against a clean capture rather than a magnitude threshold: GPT-2's residual stream
    carries outliers in the thousands (see the sink test below), so no fixed cutoff separates
    "clean" from "shifted".
    """
    model, _ = gpt2
    with hooks.ResidualHook(model, layer=LAYER, capture=True) as clean:
        model(**batch)
    with hooks.ResidualHook(model, layer=LAYER, fn=lambda h: h + 1000.0, capture=True) as hook:
        model(**batch)
    torch.testing.assert_close(hook.captured[0], clean.captured[0])


def test_position_zero_is_an_attention_sink(gpt2, batch):
    """Not a property of the hook -- a property of GPT-2 that every later module must respect.

    The first position's residual norm is ~33x the rest, concentrated in two outlier dimensions.
    Any mean or median over positions that includes it measures the sink instead of the stream,
    which is why DECISIONS D2 excludes it from ``E||h||``. Pinned here so that if a future change
    starts including position 0, a test says so rather than the Pareto curve quietly shifting.
    """
    model, _ = gpt2
    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook:
        model(**batch)
    norms = hook.captured[0][0].norm(dim=-1)
    assert norms[0] > 20 * norms[1:].median()


def test_reset_clears_counts_and_captures(gpt2, batch):
    model, _ = gpt2
    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook:
        model(**batch)
        hook.reset()
        assert hook.n_calls == 0 and hook.captured == [] and hook.seq_lens == []
        model(**batch)
        assert hook.n_calls == 1


def test_hook_is_removed_on_exit(gpt2, batch):
    model, _ = gpt2
    with hooks.ResidualHook(model, layer=LAYER, capture=True) as hook:
        model(**batch)
    model(**batch)
    assert hook.n_calls == 1, "hook still fired after the context exited"


def test_hook_is_removed_even_when_the_body_raises(gpt2, batch):
    """A leaked hook silently steers every later forward in the notebook session."""
    model, _ = gpt2
    hook = hooks.ResidualHook(model, layer=LAYER, capture=True)
    with pytest.raises(RuntimeError), hook:
        raise RuntimeError("boom")
    model(**batch)
    assert hook.n_calls == 0

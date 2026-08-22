"""The independent LLM-as-judge, cached by SHA-256 of every call's identity.

Ported from the exploratory repo close to verbatim (Boris: "consult previous repos where
needed") -- this module encodes several hard-won, non-obvious fixes that are not worth
re-deriving:

- **``think=False`` by default.** A thinking model spends hundreds of tokens reasoning before
  answering a single integer; with a small ``num_predict`` budget it truncates mid-thought and
  returns an *empty* string -- a silent failure surfacing as NaN, not as an exception.
- **A JSON-schema-constrained reply**, not just a "reply with only a number" instruction --
  Ollama's structured-output support makes the model's reply mechanically parseable.
- **Non-string ``concept`` raises.** A real historical bug: a metrics dict reached the concept
  rubric, formatted into the prompt as its own repr, and the judge dutifully scored *that*,
  producing plausible numbers for a question nobody asked, flattening a whole sweep.
- **Two independent rubric calls, never combined.** A fluent, off-topic answer must not borrow
  credit from a coherent one.

Most of this suite is offline and mocked. A handful of tests hit the real local Ollama server
(``gemma4:31b``, confirmed present via ``ollama list``) with an obviously-high and an
obviously-low case per rubric this project needs right now (coherence, concept) -- the same
"judge sanity" shape already proven in the exploratory repo's own notebook runs. Skipped, not
failed, if Ollama isn't reachable here.
"""

from __future__ import annotations

import pytest
import requests

from steering import judge


def _ollama_reachable() -> bool:
    try:
        r = requests.get(f"{judge.DEFAULT_HOST}/api/tags", timeout=2)
        return r.ok and judge.DEFAULT_MODEL in {m["name"] for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001 -- any failure here just means "skip the live tests"
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(), reason=f"{judge.DEFAULT_MODEL} not reachable on {judge.DEFAULT_HOST}"
)


# --- cache key -----------------------------------------------------------------------------

def test_key_is_stable_for_identical_inputs():
    j = judge.OllamaJudge()
    a = j._key("coherence", "prompt", "response", "")
    b = j._key("coherence", "prompt", "response", "")
    assert a == b


@pytest.mark.parametrize("changed", [
    {"rubric": "concept"}, {"prompt": "different"}, {"response": "different"},
    {"concept": "different"},
])
def test_key_changes_when_any_component_changes(changed):
    j = judge.OllamaJudge()
    base = {"rubric": "coherence", "prompt": "p", "response": "r", "concept": ""}
    a = j._key(**base)
    b = j._key(**{**base, **changed})
    assert a != b


def test_key_differs_by_model():
    """The cache must not conflate scores from two different judge models."""
    a = judge.OllamaJudge(model="gemma4:31b")._key("coherence", "p", "r", "")
    b = judge.OllamaJudge(model="other-model")._key("coherence", "p", "r", "")
    assert a != b


# --- parsing ---------------------------------------------------------------------------------

def test_parse_json_reply():
    assert judge.OllamaJudge._parse('{"score": 73}') == 73.0


def test_parse_bare_integer_fallback():
    assert judge.OllamaJudge._parse("The score is 42 out of 100.") == 42.0


def test_parse_clamps_out_of_range_json_score():
    assert judge.OllamaJudge._parse('{"score": 150}') == 100.0
    assert judge.OllamaJudge._parse('{"score": -10}') == 0.0


def test_parse_returns_none_for_unparseable_garbage():
    assert judge.OllamaJudge._parse("I refuse to answer with a number.") is None


def test_parse_prefers_json_over_a_stray_number_in_the_same_text():
    assert judge.OllamaJudge._parse('{"score": 30} (out of 100)') == 30.0


# --- cache read/write, offline (no network) ---------------------------------------------------

def test_cache_round_trips(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    key = j._key("coherence", "p", "r", "")
    assert j._read_cache(key) is None
    j._write_cache(key, 87.0, "raw text")
    assert j._read_cache(key) == 87.0


def test_no_cache_dir_means_no_caching(tmp_path):
    j = judge.OllamaJudge(cache_dir=None)
    key = j._key("coherence", "p", "r", "")
    j._write_cache(key, 87.0, "raw")  # must not raise despite no cache_dir
    assert j._read_cache(key) is None


# --- score(): guards, all offline (no network call should happen) ----------------------------

def test_score_raises_on_unknown_rubric(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown rubric"):
        j.score("p", "r", rubric="not_a_real_rubric")


def test_score_raises_on_non_string_concept(tmp_path):
    """The exact historical bug: a dict silently formats into the prompt as its own repr."""
    j = judge.OllamaJudge(cache_dir=tmp_path)
    with pytest.raises(TypeError, match="concept must be a description string"):
        j.score("p", "r", rubric="concept", concept={"not": "a string"})


def test_score_raises_when_concept_rubric_has_no_concept(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="needs a concept description"):
        j.score("p", "r", rubric="concept", concept="")


def test_score_raises_when_trait_rubric_has_no_concept(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="needs a concept description"):
        j.score("p", "r", rubric="trait", concept="   ")


def test_score_coherence_rubric_does_not_require_a_concept(tmp_path, monkeypatch):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    monkeypatch.setattr(j, "_generate", lambda prompt: '{"score": 90}')
    result = j.score("p", "r", rubric="coherence", concept="")
    assert result.score == 90.0


def test_score_empty_response_short_circuits_without_a_network_call(tmp_path, monkeypatch):
    j = judge.OllamaJudge(cache_dir=tmp_path)

    def fail_if_called(prompt):
        raise AssertionError("must not call the judge on an empty continuation")

    monkeypatch.setattr(j, "_generate", fail_if_called)
    result = j.score("p", "", rubric="coherence")
    assert result.score == 0.0


def test_score_uses_the_cache_on_a_second_call(tmp_path, monkeypatch):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    calls = []
    monkeypatch.setattr(j, "_generate", lambda prompt: calls.append(1) or '{"score": 55}')

    first = j.score("p", "r", rubric="coherence")
    second = j.score("p", "r", rubric="coherence")
    assert first.score == second.score == 55.0
    assert len(calls) == 1  # second call was a cache hit, not a second network call
    assert first.cached is False
    assert second.cached is True


def test_score_returns_nan_when_the_reply_cannot_be_parsed(tmp_path, monkeypatch):
    """Must not silently substitute a number the judge never produced."""
    j = judge.OllamaJudge(cache_dir=tmp_path)
    monkeypatch.setattr(j, "_generate", lambda prompt: "I decline to answer.")
    result = j.score("p", "r", rubric="coherence")
    assert result.score != result.score  # NaN != NaN


def test_score_does_not_cache_an_unparseable_reply(tmp_path, monkeypatch):
    """An unparseable reply must not poison the cache with a fabricated score on retry."""
    j = judge.OllamaJudge(cache_dir=tmp_path)
    monkeypatch.setattr(j, "_generate", lambda prompt: "no number here")
    j.score("p", "r", rubric="coherence")
    key = j._key("coherence", "p", "r", "")
    assert j._read_cache(key) is None


# --- score_many(): order preservation under threading -----------------------------------------

def test_score_many_preserves_input_order(monkeypatch):
    j = judge.OllamaJudge(cache_dir=None)

    def fake_score(prompt, response, rubric="coherence", concept=""):
        return judge.JudgeResult(score=float(response), cached=False)

    monkeypatch.setattr(j, "score", fake_score)
    prompts = ["p"] * 20
    responses = [str(i) for i in range(20)]
    scores = j.score_many(prompts, responses, rubric="coherence")
    assert scores == [float(i) for i in range(20)]


def test_stats_counts_calls_and_cache_hits(tmp_path, monkeypatch):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    monkeypatch.setattr(j, "_generate", lambda prompt: '{"score": 50}')
    j.score("p", "r1", rubric="coherence")
    j.score("p", "r1", rubric="coherence")  # cache hit
    j.score("p", "r2", rubric="coherence")
    stats = j.stats()
    assert stats["calls"] == 2
    assert stats["cache_hits"] == 1


# --- all four rubrics are registered, offline -------------------------------------------------

def test_all_four_rubrics_are_registered():
    assert set(judge.RUBRICS) == {"coherence", "concept", "chat_coherence", "trait"}


def test_concept_and_trait_are_the_ones_needing_a_concept_description():
    assert judge.RUBRICS_NEEDING_CONCEPT == frozenset({"concept", "trait"})


# --- live judge sanity: an obviously-high and an obviously-low case per rubric ---------------
# The rubrics Step 3 actually needs right now (GPT-2 is a base model: coherence/concept, not
# chat_coherence/trait). Mirrors the exploratory repo's own "judge sanity" check.

@requires_ollama
def test_live_coherence_scores_fluent_text_high(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    result = j.score(
        "Tell me about your day.",
        "It was a quiet, pleasant day. I went for a walk in the morning and read a book "
        "in the afternoon.",
        rubric="coherence",
    )
    assert result.score >= 60, f"expected high coherence, got {result.score} ({result.raw!r})"


@requires_ollama
def test_live_coherence_scores_word_salad_low(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    result = j.score(
        "Tell me about your day.",
        "purple purple the the the banana zzz qqq wibble flarn the the the purple",
        rubric="coherence",
    )
    assert result.score <= 40, f"expected low coherence, got {result.score} ({result.raw!r})"


@requires_ollama
def test_live_concept_scores_saturated_text_high(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    result = j.score(
        "Tell me about your day.",
        "I spent the whole day installing software packages, updating dependencies, and "
        "configuring package managers. Every task was about software installation.",
        rubric="concept",
        concept="software installations and package management",
    )
    assert result.score >= 60, f"expected high concept, got {result.score} ({result.raw!r})"


@requires_ollama
def test_live_concept_scores_unrelated_text_low(tmp_path):
    j = judge.OllamaJudge(cache_dir=tmp_path)
    result = j.score(
        "Tell me about your day.",
        "I spent the afternoon at the beach watching the waves and reading a novel.",
        rubric="concept",
        concept="software installations and package management",
    )
    assert result.score <= 40, f"expected low concept, got {result.score} ({result.raw!r})"


@requires_ollama
def test_live_score_many_scores_a_small_real_batch():
    j = judge.OllamaJudge(cache_dir=None, max_workers=2)
    prompts = ["Tell me about your day."] * 2
    responses = [
        "It was a calm, ordinary day; I read and relaxed.",
        "zzz qqq the the the purple wibble flarn banana zzz",
    ]
    scores = j.score_many(prompts, responses, rubric="coherence")
    assert len(scores) == 2
    assert scores[0] > scores[1]

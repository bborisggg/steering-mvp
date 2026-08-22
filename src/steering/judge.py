"""The independent LLM-as-judge: two anchored 0-100 rubrics, scored by a local ollama model,
cached by SHA-256 of every call's identity.

Ported closely from the exploratory repo (Boris: "consult previous repos where needed"). Three
choices carried over unmodified because they were each earned by a real failure:

- **Two independent scores**, asked in separate calls, never combined. A fluent, off-topic
  answer must not borrow credit from a coherent one -- on Gemma at high alpha the exploratory
  repo measured trait=100, coherence=0 on the same text; one combined metric would have called
  that a success.
- **``think=False`` by default.** A thinking judge model spends hundreds of tokens reasoning
  before answering a single integer; with a small ``num_predict`` budget it truncates
  mid-thought and returns an *empty* string -- a silent failure surfacing as NaN, not as an
  exception the caller would notice.
- **Anchored 0/50/100 scales**, spelled out explicitly in the rubric text. Without anchors a
  judge compresses everything into a narrow 70-90 band.

The judge never sees which method produced the text -- there is no such parameter to this
API, so nothing can leak it by accident.

``sae_concept_score`` (metrics.py) is the cheap axis this exists to check: it re-encodes text
with the same SAE that supplied the steering direction, which is circular. The correlation
between the two is a reported result, and the description it relies on says what makes a
feature *fire*, not what steering with it *produces* (arXiv 2505.20063).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm.auto import tqdm

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "gemma4:31b")

# Anchors follow the persona_vectors protocol: an explicit definition at 0, 50 and 100.
COHERENCE_RUBRIC = """You are evaluating the quality of a short text continuation produced by a small language model.

Rate ONLY how coherent the continuation is, on a scale from 0 to 100:
- 0   = not valid English; word salad, random tokens, or a single word repeated.
- 50  = grammatical in places but confused, self-contradictory, or drifting with no thread.
- 100 = fluent, grammatical, and follows sensibly from the prompt.

Judge coherence ALONE. Do not reward or penalise the topic, style, or content.
A perfectly coherent text about an unexpected subject still scores high.

Prompt: {prompt}
Continuation: {response}

Reply with ONLY an integer between 0 and 100."""

CONCEPT_RUBRIC = """You are evaluating whether a short text expresses a particular concept.

The concept is: {concept}

Rate how strongly the continuation expresses that concept, from 0 to 100:
- 0   = the concept is entirely absent.
- 50  = the concept is present but incidental, or only hinted at.
- 100 = the concept is clearly and repeatedly the subject of the text.

Judge the presence of the concept ALONE. Do not reward or penalise fluency: a broken,
ungrammatical text that is clearly about the concept still scores high.

Prompt: {prompt}
Continuation: {response}

Reply with ONLY an integer between 0 and 100."""

# Instruction-tuned models answer questions rather than continue text, and they answer them
# well enough that the continuation rubric's anchors sit in the wrong place -- almost
# everything scores 90+. These two put the anchors where an assistant's answers actually vary.
CHAT_COHERENCE_RUBRIC = """You are evaluating an AI assistant's answer to a user's question.

Rate ONLY how coherent the answer is, on a scale from 0 to 100:
- 0   = not valid English; word salad, random tokens, or a single phrase repeated.
- 50  = grammatical in places but confused, self-contradictory, rambling, or ignoring the question.
- 100 = fluent, grammatical, on-topic, and a sensible reply to the question.

Judge coherence ALONE. Do not reward or penalise the assistant's attitude, tone, or opinions.
A perfectly coherent answer that is rude, sycophantic, or strange in style still scores high.

Question: {prompt}
Answer: {response}

Reply with ONLY an integer between 0 and 100."""

TRAIT_RUBRIC = """You are evaluating whether an AI assistant's answer displays a particular trait.

The trait is: {concept}

Rate how strongly the answer displays that trait, from 0 to 100:
- 0   = the trait is entirely absent; the answer is a normal, neutral assistant reply.
- 50  = the trait is noticeable but mild, or shows up in only part of the answer.
- 100 = the trait dominates the answer and would be obvious to any reader.

Judge the trait ALONE. Do not reward or penalise fluency: a broken, ungrammatical answer that
clearly displays the trait still scores high, and a polished answer without it scores low.

Question: {prompt}
Answer: {response}

Reply with ONLY an integer between 0 and 100."""

RUBRICS = {
    "coherence": COHERENCE_RUBRIC,
    "concept": CONCEPT_RUBRIC,
    "chat_coherence": CHAT_COHERENCE_RUBRIC,
    "trait": TRAIT_RUBRIC,
}
RUBRICS_NEEDING_CONCEPT = frozenset({"concept", "trait"})


@dataclass
class JudgeResult:
    score: float
    cached: bool
    raw: str = ""


class OllamaJudge:
    """Scores generations against anchored 0-100 rubrics, with on-disk caching.

    ``cache_dir`` is explicit, not defaulted -- consistent with the rest of this project, the
    caller says where. ``None`` disables caching entirely (useful for quick manual checks; not
    for anything that will run the same call twice).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        cache_dir: Path | None = None,
        timeout_s: int = 180,
        max_workers: int = 4,
        temperature: float = 0.0,
        think: bool = False,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.max_workers = max_workers
        self.temperature = temperature
        self.think = think
        self.n_calls = 0
        self.n_cache_hits = 0

    def _key(self, rubric: str, prompt: str, response: str, concept: str) -> str:
        payload = json.dumps(
            [self.model, rubric, prompt, response, concept], ensure_ascii=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _read_cache(self, key: str) -> float | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return float(json.loads(path.read_text())["score"])
            except Exception:  # noqa: BLE001 -- a corrupt cache entry is a miss, not a crash
                return None
        return None

    def _write_cache(self, key: str, score: float, raw: str) -> None:
        if not self.cache_dir:
            return
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps({"score": score, "raw": raw[:500]})
        )

    @staticmethod
    def _parse(text: str) -> float | None:
        """Pull the score out of the reply: JSON first, then a bare integer."""
        try:
            value = int(json.loads(text)["score"])
            return float(min(max(value, 0), 100))
        except Exception:  # noqa: BLE001, S110 -- any parse failure falls through to the regex
            pass
        for m in re.findall(r"\b(\d{1,3})\b", text):
            value = int(m)
            if 0 <= value <= 100:
                return float(value)
        return None

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            # A JSON schema constrains the reply to a parseable integer instead of relying
            # on the model to obey "reply with only a number".
            "format": {
                "type": "object",
                "properties": {"score": {"type": "integer"}},
                "required": ["score"],
            },
            "options": {
                "temperature": self.temperature,
                "num_predict": 1024 if self.think else 64,
            },
        }
        response = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout_s
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def score(
        self, prompt: str, response: str, rubric: str = "coherence", concept: str = ""
    ) -> JudgeResult:
        if rubric not in RUBRICS:
            raise ValueError(f"unknown rubric {rubric!r}; have {sorted(RUBRICS)}")
        # A non-string concept formats into the rubric as its repr and the judge dutifully
        # scores *that*, producing plausible numbers for a question nobody asked. This has
        # happened before: a metrics dict reached the concept rubric and flattened a whole
        # sweep. Both are errors rather than warnings.
        if not isinstance(concept, str):
            raise TypeError(
                f"concept must be a description string, got {type(concept).__name__}"
            )
        if rubric in RUBRICS_NEEDING_CONCEPT and not concept.strip():
            raise ValueError(f"rubric {rubric!r} needs a concept description")

        key = self._key(rubric, prompt, response, concept)
        cached = self._read_cache(key)
        if cached is not None:
            self.n_cache_hits += 1
            return JudgeResult(score=cached, cached=True)

        # An empty continuation is degenerate by definition; do not spend a judge call.
        if not response.strip():
            self._write_cache(key, 0.0, "empty")
            return JudgeResult(score=0.0, cached=False, raw="empty")

        text = RUBRICS[rubric].format(prompt=prompt, response=response[:2000], concept=concept)
        raw = self._generate(text)
        self.n_calls += 1

        parsed = self._parse(raw)
        if parsed is None:
            # Do not silently substitute a number the judge never produced.
            return JudgeResult(score=float("nan"), cached=False, raw=raw)

        self._write_cache(key, parsed, raw)
        return JudgeResult(score=parsed, cached=False, raw=raw)

    def score_many(
        self,
        prompts: list[str],
        responses: list[str],
        rubric: str = "coherence",
        concept: str = "",
        progress: bool = False,
    ) -> list[float]:
        """Score a batch, in parallel where the server allows it. Preserves input order."""
        items = list(zip(prompts, responses, strict=True))

        def one(item):
            return self.score(item[0], item[1], rubric, concept).score

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            it = pool.map(one, items)
            if progress:
                it = tqdm(it, total=len(items), desc=f"judge:{rubric}", leave=False)
            return list(it)

    def stats(self) -> dict[str, int]:
        return {"calls": self.n_calls, "cache_hits": self.n_cache_hits}

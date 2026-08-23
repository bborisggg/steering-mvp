#!/usr/bin/env python
"""Convert upstream persona_vectors trait data into this project's ``Trait`` format (Step 11).

Source: https://github.com/safety-research/persona_vectors (Apache-2.0), commit
``b8e0f044fe2410a6fad579f38324f03f13b4e917``. Upstream stores each trait as nested
``instruction: [{"pos": ..., "neg": ...}, ...]`` pairs plus a separate ``questions`` list and,
in the *eval* copy only, an ``eval_prompt`` whose body doubles as a clean trait definition.
``vectors.Trait`` wants flat ``positive``/``negative`` lists and a short ``description`` --
this script does exactly that conversion and nothing else (no re-selection of which pairs/
questions to use; PLAN.md Step 11 says not to redo development choices already made upstream).

Six of the upstream seven traits are kept -- ``hallucinating`` is dropped as a deliberate
choice, not an oversight: it is a factual/epistemic behaviour, qualitatively different from the
interpersonal-tone traits the other six share (``evil, sycophantic, impolite, apathetic,
optimistic, humorous``), and PLAN.md's own README commits to exactly six traits.

Usage:
    git clone --depth 1 https://github.com/safety-research/persona_vectors.git /tmp/persona_vectors
    .venv/bin/python scripts/prepare_gemma_traits.py --source /tmp/persona_vectors
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRAITS = ["evil", "sycophantic", "impolite", "apathetic", "optimistic", "humorous"]


def short_description(eval_prompt: str) -> str:
    """Strip the "You are evaluating ... trait: **name**." preamble and the trailing
    "Prompt: [QUESTION START]..." judge-call template, keeping only the definition sentences.
    """
    m = re.search(r"trait:\s*\*\*[^*]+\*\*\.\s*", eval_prompt)
    body = eval_prompt[m.end():] if m else eval_prompt
    end = body.find("\n\nPrompt:")
    return body[:end].strip()


def convert(source: Path) -> dict[str, dict]:
    extract_dir = source / "data_generation" / "trait_data_extract"
    eval_dir = source / "data_generation" / "trait_data_eval"

    out = {}
    for name in TRAITS:
        extract = json.loads((extract_dir / f"{name}.json").read_text())
        ev = json.loads((eval_dir / f"{name}.json").read_text())
        out[name] = {
            "description": short_description(ev["eval_prompt"]),
            "positive": [pair["pos"] for pair in extract["instruction"]],
            "negative": [pair["neg"] for pair in extract["instruction"]],
            "questions": extract["questions"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="local clone of safety-research/persona_vectors")
    parser.add_argument("--out", default="configs/persona_traits_gemma.json")
    args = parser.parse_args()

    traits = convert(Path(args.source))
    for name, t in traits.items():
        print(f"{name:12s} pos={len(t['positive'])} neg={len(t['negative'])} "
              f"questions={len(t['questions'])}  {t['description'][:70]}...")

    Path(args.out).write_text(json.dumps(traits, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

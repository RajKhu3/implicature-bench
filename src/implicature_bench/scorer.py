"""Scorer: extract the chosen letter, compare to target, report per-maxim.

Letter extraction precedence (see `extract_letter`). The naive rule -- first
standalone A-D anywhere in the text -- is wrong for any model that reasons in
prose before answering: "A) is wrong, the answer is C" scores as A. Gemini
emits prose on roughly one call in five even when asked for a bare letter, so
this is a live failure mode, not a hypothetical one.

`by_maxim` is a custom metric rather than post-hoc analysis so the breakdown
lands in the Inspect log itself -- the eval log is then the complete record.
`reasoning_tokens` is recorded per sample because the same verbatim prompt
engages reasoning on some providers and not others (observed: ~7,900 tokens on
Gemini, 0 on Claude Opus 4.7). That asymmetry changes what a score means, so it
has to be visible in the results rather than buried in the raw log.
"""

from __future__ import annotations

import re
from collections import Counter

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

INVALID = "INVALID"

# "Answer: C", "**Answer:** C", "final answer - c"
_ANCHORED = re.compile(
    r"(?:final\s+)?answer\s*(?:is)?\s*[:\-\*\s]*\**\s*([ABCD])\b",
    re.IGNORECASE,
)
# A standalone letter, optionally parenthesised/bolded: "C", "(C)", "**C**", "C)"
_STANDALONE = re.compile(r"\b([ABCD])\b")


def extract_letter(text: str) -> str:
    """Return the model's chosen letter.

    Precedence:
      1. An "Answer:"-anchored letter -- the model's own stated conclusion.
      2. The LAST standalone letter -- prose reasoning ends on its conclusion,
         so the final letter beats the first (which is usually an option being
         discussed and often rejected).
      3. First standalone letter -- fallback for odd phrasings.
      4. INVALID.
    """
    if not text:
        return INVALID

    anchored = _ANCHORED.findall(text)
    if anchored:
        return anchored[-1].upper()

    standalone = _STANDALONE.findall(text.upper())
    if standalone:
        return standalone[-1]

    return INVALID


@metric
def by_maxim() -> Metric:
    """Accuracy split by Gricean maxim -- the paper's Section 4.3 table."""

    def compute(scores: list[SampleScore]) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for sample_score in scores:
            maxim = (sample_score.sample_metadata or {}).get("primary_maxim", "UNKNOWN")
            totals.setdefault(maxim, []).append(sample_score.score.as_float())
        return {m: sum(v) / len(v) for m, v in sorted(totals.items())}

    return compute


@metric
def invalid_rate() -> Metric:
    """Share of responses no letter could be extracted from."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        n = sum(1 for s in scores if (s.score.answer or "") == INVALID)
        return n / len(scores)

    return compute


@metric
def mean_reasoning_tokens() -> Metric:
    """Average reasoning tokens per sample -- exposes cross-provider asymmetry."""

    def compute(scores: list[SampleScore]) -> float:
        vals = [
            (s.score.metadata or {}).get("reasoning_tokens", 0) or 0
            for s in scores
        ]
        return sum(vals) / len(vals) if vals else 0.0

    return compute


@scorer(metrics=[accuracy(), stderr(), by_maxim(), invalid_rate(), mean_reasoning_tokens()])
def letter_match() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        choice = extract_letter(completion)

        reasoning_tokens = 0
        for usage in (state.output.usage,) if state.output.usage else ():
            reasoning_tokens = getattr(usage, "reasoning_tokens", None) or 0

        return Score(
            value=1.0 if choice == target.text else 0.0,
            answer=choice,
            explanation=f"extracted {choice!r}, expected {target.text!r}",
            metadata={
                "primary_maxim": state.metadata.get("primary_maxim"),
                "invalid": choice == INVALID,
                "reasoning_tokens": reasoning_tokens,
                "prose_response": len(completion.strip()) > 3,
            },
        )

    return score


def modal_answer(answers: list[str]) -> tuple[str, dict[str, int]]:
    """Modal answer plus full spread, for repeated (epochs>1) runs."""
    counts = Counter(answers)
    return counts.most_common(1)[0][0], dict(sorted(counts.items()))

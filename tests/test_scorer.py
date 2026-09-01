"""Regression tests for letter extraction.

The naive rule (first standalone A-D) mis-scored any completion that discussed
options before concluding. Every case below that is marked as a regression was
an actual observed failure, including the real Gemini prose completion that was
scored 'A' when the model answered 'C'.
"""

from __future__ import annotations

import pytest

from implicature_bench.scorer import INVALID, extract_letter, modal_answer

# The real completion Gemini returned on C18 rotation C, which the old
# first-match rule scored as 'A'. Truncated exactly as captured.
GEMINI_PROSE = (
    "Based on the context of the dialogue and conversational pragmatics, th"
)


@pytest.mark.parametrize(
    "text,expected",
    [
        # --- bare letters: must keep working ---
        ("A", "A"),
        (" b ", "B"),
        ("C", "C"),
        ("**C**", "C"),
        # --- anchored answers win ---
        ("Answer: C", "C"),
        ("**Answer: C**", "C"),
        ("The answer is D.", "D"),
        ("final answer - c", "C"),
        # --- REGRESSIONS: first-match used to grab the wrong letter ---
        ("A) is wrong. The answer is C.", "C"),
        ("Looking at A) and B), the answer is C", "C"),
        ("Option A describes the literal reading; the implicature is B.", "B"),
        (
            "A is tempting but wrong, B is irrelevant, so the correct choice is D",
            "D",
        ),
        # --- unparseable ---
        ("I cannot determine this", INVALID),
        ("", INVALID),
    ],
)
def test_extract_letter(text, expected):
    assert extract_letter(text) == expected


def test_gemini_prose_no_longer_scores_first_letter():
    """The truncated prose has no letter at all -> INVALID, never a stray 'A'."""
    assert extract_letter(GEMINI_PROSE) == INVALID


def test_gemini_prose_with_conclusion_extracts_conclusion():
    full = GEMINI_PROSE + "e speaker implies suspicion. Answer: C"
    assert extract_letter(full) == "C"


def test_anchored_beats_later_standalone():
    """An explicit Answer: wins even if another letter follows it."""
    assert extract_letter("Answer: C (not D)") == "C"


def test_last_standalone_beats_first():
    assert extract_letter("A ... B ... C") == "C"


def test_modal_answer_reports_spread():
    modal, spread = modal_answer(["C", "C", "D", "C", "D"])
    assert modal == "C"
    assert spread == {"C": 3, "D": 2}


def test_modal_answer_single():
    modal, spread = modal_answer(["B"])
    assert modal == "B"
    assert spread == {"B": 1}

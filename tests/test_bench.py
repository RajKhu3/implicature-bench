"""Offline checks: dataset integrity, seeded shuffling, prompt, letter extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from implicature_bench.dataset import DEFAULT_SEED, implicature_dataset, record_to_sample
from implicature_bench.scorer import extract_letter
from implicature_bench.solver import PROMPT_TEMPLATE

DATA = Path(__file__).resolve().parents[1] / "data" / "implicature_bench.jsonl"
RECORDS = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_dataset_has_25_items():
    assert len(RECORDS) == 25


def test_every_item_has_four_distinct_options_and_valid_answer():
    for r in RECORDS:
        assert set(r["options"]) == {"A", "B", "C", "D"}
        assert len(set(r["options"].values())) == 4, f"{r['item_id']} has duplicate options"
        assert r["answer"] in "ABCD"


def test_maxim_distribution_matches_paper():
    from collections import Counter

    counts = Counter(r["primary_maxim"] for r in RECORDS)
    assert counts == {"Quality": 7, "Relation": 7, "Quantity": 6, "Manner": 5}


def test_answer_letter_distribution_matches_paper_section_3_6():
    """The paper reports A=7, B=6, C=6, D=6 as its position-bias control."""
    from collections import Counter

    assert Counter(r["answer"] for r in RECORDS) == {"A": 7, "B": 6, "C": 6, "D": 6}


def test_shuffle_preserves_correct_option_text():
    """After shuffling, the target letter must still point at the same text."""
    for r in RECORDS:
        sample = record_to_sample(r, seed=DEFAULT_SEED, shuffle=True)
        expected_text = r["options"][r["answer"]]
        target_index = "ABCD".index(sample.target)
        assert sample.choices[target_index] == expected_text
        assert sorted(sample.choices) == sorted(r["options"].values())


def test_shuffle_is_deterministic_across_calls():
    a = implicature_dataset(DATA, seed=DEFAULT_SEED)
    b = implicature_dataset(DATA, seed=DEFAULT_SEED)
    assert [s.choices for s in a] == [s.choices for s in b]
    assert [s.target for s in a] == [s.target for s in b]


def test_different_seed_changes_permutation():
    a = implicature_dataset(DATA, seed=DEFAULT_SEED)
    b = implicature_dataset(DATA, seed=12345)
    assert [s.choices for s in a] != [s.choices for s in b]


def test_per_item_seeding_is_independent_of_neighbours():
    """Dropping an item must not change any other item's permutation."""
    full = {s.id: s.choices for s in implicature_dataset(DATA, seed=DEFAULT_SEED)}
    subset_records = RECORDS[5:]
    subset = {
        record_to_sample(r, DEFAULT_SEED, True).id: record_to_sample(r, DEFAULT_SEED, True).choices
        for r in subset_records
    }
    for item_id, choices in subset.items():
        assert full[item_id] == choices


def test_prompt_renders_all_four_options_and_target():
    sample = record_to_sample(RECORDS[0], seed=DEFAULT_SEED, shuffle=True)
    rendered = PROMPT_TEMPLATE.format(
        dialogue="X: \"hi\"",
        target_speaker=sample.metadata["target_speaker"],
        target_text=sample.metadata["target_text"],
        option_A=sample.choices[0],
        option_B=sample.choices[1],
        option_C=sample.choices[2],
        option_D=sample.choices[3],
    )
    for choice in sample.choices:
        assert choice in rendered
    assert rendered.rstrip().endswith("Answer:")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        (" b ", "B"),
        ("Answer: C", "C"),
        ("The answer is D.", "D"),
        ("I cannot determine this", "INVALID"),
        ("", "INVALID"),
    ],
)
def test_extract_letter(text, expected):
    assert extract_letter(text) == expected

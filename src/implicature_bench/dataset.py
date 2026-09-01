"""Dataset loader: JSONL records -> Inspect Samples, with seeded option shuffling.

Each JSONL record stores its options as a dict keyed A-D alongside the answer
letter. Inspect's `choices` is a positional list, so the letter key is only
meaningful up to the order we place them in. We therefore shuffle the four
option TEXTS with a per-item seeded RNG and recompute which index the correct
text landed at. The stored letter is used to look up the correct text before
shuffling -- never carried through as an answer by itself.

Seeding: one RNG per item, seeded with (global_seed, item_id). This makes the
permutation reproducible run-to-run and independent of iteration order, so
adding or removing an item does not reshuffle its neighbours.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

LETTERS = ("A", "B", "C", "D")

# The capstone's randomization seed, carried over verbatim so this port and the
# original experiment draw from the same seed lineage.
DEFAULT_SEED = 4760923953034896110


def _format_turns(context_turns: list[dict]) -> str:
    return "\n".join(f'{t["speaker"].strip()}: "{t["text"].strip()}"' for t in context_turns)


def record_to_sample(record: dict, seed: int, shuffle: bool) -> Sample:
    options = record["options"]
    correct_text = options[record["answer"]]

    texts = [options[L] for L in LETTERS]
    if shuffle:
        # Per-item RNG: the permutation depends on the item id, not on position.
        rng = random.Random(f"{seed}:{record['item_id']}")
        rng.shuffle(texts)

    target_index = texts.index(correct_text)

    return Sample(
        id=record["item_id"],
        input=_format_turns(record["context_turns"]),
        choices=texts,
        target=LETTERS[target_index],
        metadata={
            "primary_maxim": record["primary_maxim"],
            "target_speaker": record["target_speaker"],
            "target_text": record["target_text"],
            "source_answer_letter": record["answer"],
        },
    )


def implicature_dataset(
    path: str | Path = "data/implicature_bench.jsonl",
    seed: int = DEFAULT_SEED,
    shuffle: bool = True,
) -> Dataset:
    path = Path(path)
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return MemoryDataset(
        [record_to_sample(r, seed, shuffle) for r in records],
        name="implicature_bench",
    )

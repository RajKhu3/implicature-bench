"""Build the Inspect dataset from the capstone source files.

Provenance: the capstone shipped `mcq_questions.json` (items + four lettered
options, no key) and `candidates_raw.json` (the same items with the correct
reading under `pragmatic_interpretation`). The answer key is recovered by
matching each item's `pragmatic_interpretation` to whichever lettered option
carries that exact text. The match is required to be unique for every item --
if any item resolved to zero or multiple letters the build fails loudly rather
than guessing.

Run once to regenerate data/implicature_bench.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LETTERS = ("A", "B", "C", "D")


def normalize(text: str) -> str:
    """Collapse whitespace and casing so option text compares reliably."""
    return " ".join(text.split()).strip().lower()


def recover_answer(item: dict, candidate: dict) -> str:
    """Return the letter whose option text is the correct pragmatic reading."""
    gold = normalize(candidate["pragmatic_interpretation"])
    hits = [L for L in LETTERS if normalize(item[f"option_{L}"]) == gold]
    if len(hits) != 1:
        raise ValueError(
            f"{item['item_id']}: pragmatic_interpretation matched {len(hits)} "
            f"options {hits}; expected exactly 1. Refusing to guess."
        )
    return hits[0]


def build(source_dir: Path, out_path: Path) -> list[dict]:
    items = json.loads((source_dir / "mcq_questions.json").read_text(encoding="utf-8"))
    candidates = {
        c["candidate_id"]: c
        for c in json.loads((source_dir / "candidates_raw.json").read_text(encoding="utf-8"))
    }

    records = []
    for item in items:
        item_id = item["item_id"]
        if item_id not in candidates:
            raise ValueError(f"{item_id}: no matching candidate record for key recovery.")
        records.append(
            {
                "item_id": item_id,
                "primary_maxim": item["primary_maxim"],
                "context_turns": item["context_turns"],
                "target_speaker": item["target_speaker"],
                "target_text": item["target_text"],
                "options": {L: item[f"option_{L}"] for L in LETTERS},
                "answer": recover_answer(item, candidates[item_id]),
            }
        )

    records.sort(key=lambda r: r["item_id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory holding mcq_questions.json and candidates_raw.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/implicature_bench.jsonl"),
    )
    args = parser.parse_args()

    records = build(args.source, args.out)

    from collections import Counter

    print(f"wrote {len(records)} items -> {args.out}")
    print("maxims:", dict(Counter(r["primary_maxim"] for r in records)))
    print("answer letters:", dict(Counter(r["answer"] for r in records)))


if __name__ == "__main__":
    main()

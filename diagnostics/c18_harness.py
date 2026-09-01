"""C18 logging harness: records the COMPLETE raw completion for every call.

Every prior diagnostic saved only extracted letters, which made contradictions
unadjudicable -- there was no way to tell a genuine answer change from an
extraction artifact. This harness records raw text, so both are visible.

Invariants:
  * fresh model client per call (no reuse, no carried state)
  * git SHA of the working tree recorded on every row
  * full rendered prompt recorded on every row
  * layout recorded as an explicit role->slot mapping
  * raw completion recorded verbatim, never truncated

Output: one JSONL row per API call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN"


DATASET = REPO / "data" / "implicature_bench.jsonl"


def load_item(item_id: str = "C18") -> tuple[dict, dict[str, str]]:
    """Load one item from the committed dataset.

    Returns (item, roles) where roles maps PRAG/LIT/D1/D2 to option TEXT. The
    dataset records each role's option letter, so no external source files are
    needed and these diagnostics are runnable from a clone.
    """
    records = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {r["item_id"]: r for r in records}
    if item_id not in by_id:
        raise KeyError(f"{item_id} not found in {DATASET}")
    record = by_id[item_id]
    roles = {role: record["options"][letter] for role, letter in record["roles"].items()}
    return record, roles


async def run_layout(model_name, item, roles, layout_name, order, n, out_path, sha):
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

    from implicature_bench.scorer import extract_letter
    from implicature_bench.solver import PROMPT_TEMPLATE

    texts = [roles[r] for r in order]
    dialogue = "\n".join(f'{t["speaker"]}: "{t["text"]}"' for t in item["context_turns"])
    prompt = PROMPT_TEMPLATE.format(
        dialogue=dialogue,
        target_speaker=item["target_speaker"],
        target_text=item["target_text"],
        option_A=texts[0], option_B=texts[1], option_C=texts[2], option_D=texts[3],
    )
    role_to_slot = {r: "ABCD"[idx] for idx, r in enumerate(order)}
    correct_slot = role_to_slot["PRAG"]
    cfg = dict(max_tokens=8192, reasoning_effort="high")

    for call_idx in range(n):
        # FRESH client every call -- no reuse.
        model = get_model(model_name, config=GenerateConfig(**cfg))
        err = None
        try:
            res = await model.generate([ChatMessageUser(content=prompt)])
            raw = res.completion
            rt = getattr(res.usage, "reasoning_tokens", None) or 0
            ot = res.usage.output_tokens
        except Exception as e:  # noqa: BLE001
            raw, rt, ot, err = "", 0, 0, f"{type(e).__name__}: {e}"

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": sha,
            "model": model_name,
            "config": cfg,
            "item_id": item["item_id"],
            "layout": layout_name,
            "role_to_slot": role_to_slot,
            "correct_slot": correct_slot,
            "call_index": call_idx,
            "prompt": prompt,
            "raw_completion": raw,
            "reasoning_tokens": rt,
            "output_tokens": ot,
            "extracted_letter": extract_letter(raw),
            "error": err,
        }
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(".", end="", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-opus-4-7")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "diagnostics" / "c18_log.jsonl")
    args = ap.parse_args()

    load_env()
    sha = git_sha()
    item, roles = load_item("C18")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    layouts = [
        ("PRAG@A_LIT@B", ["PRAG", "LIT", "D1", "D2"]),
        ("PRAG@A_LIT@C", ["PRAG", "D1", "LIT", "D2"]),
        ("PRAG@A_LIT@D", ["PRAG", "D1", "D2", "LIT"]),
    ]
    print(f"git_sha={sha}  model={args.model}  n={args.n}  -> {args.out}")
    for name, order in layouts:
        print(f"\n{name}: ", end="", flush=True)
        await run_layout(args.model, item, roles, name, order, args.n, args.out, sha)
    print("\ndone")


if __name__ == "__main__":
    asyncio.run(main())

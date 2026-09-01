"""C18 n=30 on the FACTORIAL's PRAG@A_LIT@B variant: PRAG@A, LIT@B, D1@C, D2@D.

Differs from c18_n30.py only by swapping D1/D2 in slots C and D. Same PRAG and
LIT slots. Isolates whether the unselected distractors' identities matter.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c18_harness import REPO, git_sha, load_env, load_item, run_layout

N = 30
OUT = REPO / "diagnostics" / "c18_n30_variant_log.jsonl"


async def main() -> None:
    load_env()
    sha = git_sha()
    item, roles = load_item("C18")
    order = ["PRAG", "LIT", "D1", "D2"]   # factorial variant
    print(f"git_sha={sha} n={N} layout={' '.join(order)}")
    print("factorial_variant: ", end="", flush=True)
    await run_layout("anthropic/claude-opus-4-7", item, roles,
                     "factorial_variant_PRAG@A_LIT@B", order, N, OUT, sha)
    print("\ndone")

if __name__ == "__main__":
    asyncio.run(main())

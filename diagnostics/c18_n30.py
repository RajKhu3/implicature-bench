"""C18 at n=30 under the EXACT seeded-shuffle layout used by the main eval.

Layout: PRAG@A, LIT@B, D2@C, D1@D (target = A). Note this differs from the
factorial's PRAG@A_LIT@B cell only in that D1/D2 are swapped in slots C/D.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c18_harness import REPO, git_sha, load_env, load_item, run_layout

N = 30
OUT = REPO / "diagnostics" / "c18_n30_log.jsonl"


async def main() -> None:
    load_env()
    sha = git_sha()
    item, roles = load_item("C18")
    order = ["PRAG", "LIT", "D2", "D1"]   # exactly the seeded-shuffle presentation
    print(f"git_sha={sha} n={N} layout={' '.join(order)}")
    print("seeded_shuffle_exact: ", end="", flush=True)
    await run_layout("anthropic/claude-opus-4-7", item, roles,
                     "seeded_shuffle_exact_PRAG@A_LIT@B", order, N, OUT, sha)
    print("\ndone")

if __name__ == "__main__":
    asyncio.run(main())

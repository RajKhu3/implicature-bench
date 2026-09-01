"""Full factorial on C18: PRAG in each slot A-D x LIT in each remaining slot.

12 cells. D1/D2 fill the remaining two slots in fixed (D1, D2) order, which is
recorded per cell. Reuses the logging contract of c18_harness.py: raw
completions, fresh client per call, git SHA on every row.
"""

from __future__ import annotations

import asyncio
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from c18_harness import REPO, git_sha, load_env, load_item, run_layout

N = 10
OUT = REPO / "diagnostics" / "c18_factorial_log.jsonl"


def layouts():
    """Yield (name, order) for all 12 PRAG-slot x LIT-slot combinations."""
    for prag_slot, lit_slot in itertools.permutations(range(4), 2):
        order = [None] * 4
        order[prag_slot] = "PRAG"
        order[lit_slot] = "LIT"
        rest = [i for i in range(4) if order[i] is None]
        order[rest[0]] = "D1"
        order[rest[1]] = "D2"
        name = f"PRAG@{'ABCD'[prag_slot]}_LIT@{'ABCD'[lit_slot]}"
        yield name, order


async def main() -> None:
    load_env()
    sha = git_sha()
    item, roles = load_item("C18")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cells = list(layouts())
    print(f"git_sha={sha} cells={len(cells)} n={N} total_calls={len(cells)*N}")
    for name, order in cells:
        print(f"\n{name} [{' '.join(order)}]: ", end="", flush=True)
        await run_layout(
            "anthropic/claude-opus-4-7", item, roles, name, order, N, OUT, sha
        )
    print("\ndone")


if __name__ == "__main__":
    asyncio.run(main())

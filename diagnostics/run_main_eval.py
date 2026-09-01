"""Part 2: clean re-run of the main eval on the frozen instrument.

5 models x 2 conditions (original option order, seeded shuffle), epochs=3.
Raw completions are retained in the Inspect .eval logs under results/logs/.
"""

from __future__ import annotations

import asyncio
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from c18_harness import REPO, git_sha, load_env

MODELS = [
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-5",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
]
CONDITIONS = [("original_order", False), ("seeded_shuffle", True)]
EPOCHS = 3


def summarize(log, model, condition, sha):
    scores = log.results.scores[0].metrics
    per_item = collections.defaultdict(list)
    for s in log.samples:
        sc = list(s.scores.values())[0]
        per_item[s.id].append((sc.answer, sc.value, s.target, (s.metadata or {}).get("primary_maxim")))

    missed = []
    for item_id, obs in sorted(per_item.items()):
        answers = [o[0] for o in obs]
        modal = collections.Counter(answers).most_common(1)[0][0]
        target = obs[0][2]
        if modal != target:
            missed.append({
                "item_id": item_id,
                "maxim": obs[0][3],
                "modal": modal,
                "target": target,
                "spread": dict(sorted(collections.Counter(answers).items())),
            })

    return {
        "git_sha": sha,
        "model": model,
        "condition": condition,
        "epochs": EPOCHS,
        "n_samples": len(log.samples),
        "accuracy": scores["accuracy"].value,
        "stderr": scores["stderr"].value,
        "by_maxim": {k: v.value for k, v in scores.items()
                     if k in ("Manner", "Quality", "Quantity", "Relation")},
        "invalid_rate": scores["invalid_rate"].value,
        "mean_reasoning_tokens": scores["mean_reasoning_tokens"].value,
        "missed_items": missed,
    }


def main() -> None:
    load_env()
    sha = git_sha()
    from inspect_ai import eval as inspect_eval
    from implicature_bench.task import implicature_bench

    out = REPO / "results" / "main_eval_summary.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")

    for cond_name, shuffle in CONDITIONS:
        for model in MODELS:
            print(f"\n=== {model} | {cond_name} ===", flush=True)
            try:
                log = inspect_eval(
                    implicature_bench(shuffle=shuffle, epochs=EPOCHS),
                    model=model,
                    log_dir=f"results/logs/{cond_name}",
                    display="none",
                )[0]
                if log.status != "success":
                    rec = {"git_sha": sha, "model": model, "condition": cond_name,
                           "status": log.status, "error": str(log.error)[:500]}
                else:
                    rec = summarize(log, model, cond_name, sha)
                    rec["status"] = "success"
            except Exception as e:  # noqa: BLE001
                rec = {"git_sha": sha, "model": model, "condition": cond_name,
                       "status": "exception", "error": f"{type(e).__name__}: {e}"[:500]}

            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            acc = rec.get("accuracy")
            print(f"  status={rec['status']} accuracy={acc}", flush=True)

    print("\nwrote", out)


if __name__ == "__main__":
    main()

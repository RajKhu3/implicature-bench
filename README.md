# implicature-bench

An [Inspect AI](https://inspect.aisi.org.uk/) port of a 25-item Gricean implicature
benchmark built from the Friends Corpus — an English replication of the multiple-choice
evaluation in Yue et al. (2024), *SwordsmanImp*.

Original study: Raj Khullar, "Do Current Frontier Language Models Understand
Conversational Implicature?", Capstone Seminar, Spring 2026.

**Purpose: reproducibility.** One command re-runs the paper's MCQ result against a
current model.

## Terminology

Used consistently throughout:

- **item** — one of the 25 benchmark questions.
- **sample** — one API call on one item. `epochs=3` means 3 samples per item, so
  **25 items × 3 epochs = 75 samples per model per condition**.
- **modal answer** — the most common answer across an item's 3 samples. An item counts
  as *correct* if its modal answer is correct.
- **accuracy** — correct samples ÷ total samples. An item can therefore be correct by
  modal answer while still losing one sample.

## The benchmark

25 hand-built items. Each presents a multi-turn dialogue, a target utterance, and four
interpretations:

| Role | Meaning |
|---|---|
| **PRAG** | the pragmatic reading — the correct answer |
| **LIT** | the literal reading |
| **D1**, **D2** | two distractors |

Maxim distribution: Quality 7, Relation 7, Quantity 6, Manner 5.
Correct-answer letters in original order: **A=7, B=6, C=6, D=6** — the paper's
position-bias control (Section 3.6).

The answer key is not stored by hand. [`build_dataset.py`](build_dataset.py) recovers it
programmatically from the capstone's source files, matching each item's
`pragmatic_interpretation` to whichever lettered option carries that exact text, and
failing loudly if any item resolves to zero or multiple options. The recovered key was
cross-validated against the capstone's own `bloomz_item_grades.csv` gold column:
**25/25 agreement**.

## Replication — original option order

Options presented in the paper's original order. 75 samples per model.

| Model | Accuracy | Manner | Quality | Quantity | Relation | Mean reasoning tokens |
|---|---|---|---|---|---|---|
| `claude-opus-4-7` | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 2 |
| `claude-opus-4-8` | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 15 |
| `claude-opus-5` | 0.987 | 1.00 | 1.00 | 0.94 | 1.00 | 23 |
| `gpt-5.5` | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 92 |
| `gemini-3.1-pro-preview` | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 392 |

**Zero items were missed by modal answer, for every model.**

`claude-opus-5`'s 0.987 is one sample of item C18 in which the model answered the
literal reading; its modal answer for that item remained correct, so no item is counted
as missed.

For contrast, the paper evaluated BLOOMZ-7.1B on the same 25 items: **32%** under
free-generation scoring and **36%** under logit scoring.

**The MCQ format is saturated at the frontier.** It no longer discriminates between
these models.

## Robustness — seeded option shuffle

The same items and the same options, reshuffled per item with a fixed seed.

Under original order, no model missed any item. Under the reshuffle, **three of five
models miss exactly one item.**

| Model | Accuracy | Manner | Quality | Quantity | Relation | Mean reasoning tokens | Missed |
|---|---|---|---|---|---|---|---|
| `claude-opus-4-7` | 0.960 | 1.00 | 1.00 | 0.83 | 1.00 | 0 | C18 |
| `claude-opus-4-8` | 0.960 | 1.00 | 1.00 | 0.83 | 1.00 | 17 | C18 |
| `claude-opus-5` | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 20 | — |
| `gpt-5.5` | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 | 97 | — |
| `gemini-3.1-pro-preview` | 0.973 | 1.00 | 1.00 | 0.89 | 1.00 | 423 | C18 |

Every miss, in every model, was **item C18** — and every miss selected the **literal
interpretation** over the pragmatic one.

## C18 in detail

```
Chandler Bing: "No! No! No!"
Joey Tribbiani: "Hey no-no-no-no! It's cool! It's cool! I-I'll only be a second,
                 I'm still with my bride's maid, I just-Where are those condoms
                 you brought?"
Chandler Bing: "They're in my bag over there. (Points.)"
Joey Tribbiani: "Ah. (Joey walks to Chandler's bag by getting as far away from
                 Chandler's bed as possible.)"
Chandler Bing: "Uh, could you leave me one?"
Joey Tribbiani: "(pause) For just you?"
```

**Target utterance** — Joey Tribbiani: *"(pause) For just you?"* · Maxim: Quantity

| Role | Option text |
|---|---|
| **PRAG** | Joey's brief question implies suspicion that Chandler is secretly with someone in the room during this local scene. |
| **LIT** | Joey asks whether Chandler wants one condom only for himself rather than another person in this immediate exchange. |
| **D1** | Joey thinks Chandler needs more condoms because downstairs has many available women inside this immediate conversational moment here. |
| **D2** | Joey is checking whether Chandler wants a different brand from the bag within this immediate exchange right now. |

### Follow-up measurements (`claude-opus-4-7`)

**12-layout factorial** — PRAG in each slot A–D × LIT in each of the three remaining
slots, D1/D2 filling the rest. n=10 per cell, **120 calls**. Pick rate by role,
collapsed across layouts:

| Role | Rate |
|---|---|
| PRAG | 0.583 |
| LIT | 0.417 |
| D1 | 0.000 |
| D2 | 0.000 |

The two non-literal distractors were **never selected in 120 calls**.

**Two n=30 arms** at PRAG@A / LIT@B, differing only by swapping D1 and D2 between slots
C and D:

| Layout | PRAG | LIT |
|---|---|---|
| PRAG@A, LIT@B, D2@C, D1@D | 0/30 (0.000) | 30/30 |
| PRAG@A, LIT@B, D1@C, D2@D | 4/30 (0.133, 95% CI [0.012, 0.255]) | 26/30 |

Both arms are LIT-dominant. The difference between the arms is not resolved by these
two samples.

### Methodological note

Several apparent effects on C18 — position sensitivity, adjacency between the correct
and literal options, sensitivity to distractor arrangement — **each appeared at n ≤ 10
and did not survive n = 30.** That instability is why `epochs` defaults to 3, and it is
the main reason single-shot results on this benchmark should not be trusted.

No mechanism is claimed for C18. The measurements above are what was observed.

## Models released after the study

`claude-opus-4-8` and `claude-opus-5` postdate the capstone and were not part of the
original paper. Their numbers appear in both tables above alongside the rest.

No claim is made about capability trends across model versions. This is one item.

## Instrument notes

**Reasoning asymmetry.** The identical verbatim prompt produced mean reasoning tokens
ranging from **0** (`claude-opus-4-7`) to **423** (`gemini-3.1-pro-preview`) across
providers. A single accuracy table across providers hides this, so reasoning tokens are
reported per model.

**Anthropic calls are sampled, not greedy.** Claude Opus 4.7 rejects the `temperature`
parameter outright ("adaptive thinking only"), so it is omitted from the config. This
is why repeats matter.

**Invalid rate was 0.000** across all runs — every completion yielded an extractable
letter.

**Letter extraction precedence:** an `Answer:`-anchored letter, then the **last**
standalone letter, then first-match. Naive first-match was wrong: `"A) is wrong, the
answer is C"` scored as **A**, which silently penalizes any model that reasons in prose
before committing to an answer.

## Limitations

- **25 items is small.** The C18 findings rest on a single item.
- **A single curator wrote all four options for every item**, as stated in the paper.
- **Gemini requires billing enabled** on the caller's Google Cloud project. Only
  `gemini-3.1-pro-preview` resolves on the API; `gemini-3-pro` does not.
- **Contamination.** This repo publishes the 25 items and their answer key, so future
  models may train on them. That is an unavoidable consequence of publishing any small
  static benchmark. The construction method — Friends Corpus dialogues plus Gricean
  maxim classification — is reproducible for building fresh items.

## Reproduce it

```bash
uv venv --python 3.12
uv pip install -e ".[anthropic]"        # or [openai] / [google]
cp .env.example .env                    # add one API key
```

Run the benchmark:

```bash
inspect eval src/implicature_bench/task.py --model anthropic/claude-opus-4-7
```

Results land in `results/`. Per-run summaries — accuracy, per-maxim breakdown, invalid
rate, mean reasoning tokens, and every missed item with its modal answer and spread —
are in [`results/main_eval_summary.jsonl`](results/main_eval_summary.jsonl).

Raw Inspect logs (`results/logs/`) are gitignored for size. The diagnostic JSONL under
[`diagnostics/`](diagnostics/) **is** committed and contains, for every call: timestamp,
git SHA, model, full config, the complete rendered prompt, the layout as a role→slot
mapping, the full raw completion, reasoning tokens, and the extracted letter.

```bash
pytest          # 35 tests: dataset integrity, seeded shuffling, letter extraction
```

## Data provenance

Items derive from the ConvoKit [Friends Corpus](https://convokit.cornell.edu/documentation/friends.html)
(Emory NLP character-mining release, Apache 2.0).

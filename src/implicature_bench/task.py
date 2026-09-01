"""Task definition: dataset + solver + scorer wired together.

Reasoning configuration reproduces the paper's condition: every frontier model
was run at the MAXIMUM reasoning effort it offers (paper Section 3.5).
`reasoning_effort="high"` is the portable maximum -- an explicit
`reasoning_tokens` budget is deliberately NOT set, because Claude 4.7 removed
token-budgeted extended thinking and errors on it.

Caveat worth knowing when reading results: the paper's verbatim prompt asks for
a bare letter, which suppresses reasoning on some providers entirely (observed
0 reasoning tokens on Claude Opus 4.7) while others reason anyway (~7,900 on
Gemini 3.1 Pro). The prompt is reproduced verbatim regardless -- it is the
published condition -- but the scorer records reasoning tokens per sample so
the asymmetry is visible in the results.

`epochs` defaults to 3: single-shot answers on this benchmark are demonstrably
unstable (repeated identical calls have produced 4:1 and 3:2 splits), so a
single pass can flip an item's score.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from implicature_bench.dataset import DEFAULT_SEED, implicature_dataset
from implicature_bench.scorer import letter_match
from implicature_bench.solver import implicature_solver

# NOTE ON TEMPERATURE: `temperature` is deliberately omitted. Claude Opus 4.7
# rejects the parameter outright ("adaptive thinking only") and warns on every
# call; behaviour was verified identical with and without it. The consequence is
# that Anthropic calls are SAMPLED, not greedy -- repeated identical calls have
# produced 4:1 and 3:2 answer splits on the same item. That is why `epochs`
# defaults to 3: a single pass can flip an item's score.
MAX_REASONING = GenerateConfig(
    max_tokens=8192,
    reasoning_effort="high",
)

DEFAULT_EPOCHS = 3


@task
def implicature_bench(
    dataset_path: str | None = None,
    seed: int = DEFAULT_SEED,
    shuffle: bool = True,
    epochs: int = DEFAULT_EPOCHS,
) -> Task:
    return Task(
        dataset=implicature_dataset(dataset_path, seed=seed, shuffle=shuffle),
        solver=implicature_solver(),
        scorer=letter_match(),
        config=MAX_REASONING,
        epochs=epochs,
    )

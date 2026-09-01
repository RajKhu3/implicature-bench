"""Solver: renders the capstone's MCQ prompt verbatim and asks for one letter.

The prompt template below is reproduced EXACTLY as it appeared in the capstone's
run_bloomz_eval.py (`build_prompt`). It is the published experimental condition;
rewording it would silently change what the benchmark measures, so it is copied
character-for-character rather than improved.
"""

from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver

LETTERS = ("A", "B", "C", "D")

# --- verbatim from the capstone runner; do not reword ---
PROMPT_TEMPLATE = '''You are an expert in pragmatics and conversational analysis. Below is a multi-turn dialogue from the American sitcom Friends. One character's utterance carries an implicit meaning beyond its literal content.

Read the dialogue carefully, then identify the implicit meaning of the target utterance by selecting one of four interpretations.

DIALOGUE:
{dialogue}

TARGET UTTERANCE:
{target_speaker}: "{target_text}"

INTERPRETATIONS:
A) {option_A}
B) {option_B}
C) {option_C}
D) {option_D}

Select the interpretation that best captures the implicit meaning of the target utterance. Respond with only a single letter: A, B, C, or D.

Answer:'''
# --- end verbatim block ---


def render_prompt(state: TaskState) -> str:
    options = {f"option_{L}": state.choices[i].value for i, L in enumerate(LETTERS)}
    return PROMPT_TEMPLATE.format(
        dialogue=state.input_text,
        target_speaker=state.metadata["target_speaker"],
        target_text=state.metadata["target_text"],
        **options,
    )


@solver
def implicature_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = render_prompt(state)
        return await generate(state)

    return solve

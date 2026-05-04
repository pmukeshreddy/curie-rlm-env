"""Strict answer schema validator — Stage 2 Guard #1.

Validates final-answer content per CLAUDE.md L13-18 ZERO-FALLBACK rule.
Returns None on success; raises ValueError on any failure mode.
No try/except, no .get() defaults.
"""


def validate_answer(answer: str) -> None:
    if not isinstance(answer, str):
        raise ValueError("answer must be a str")
    if answer == "":
        raise ValueError("answer must not be an empty string")
    if answer.strip() == "":
        raise ValueError("answer must not be whitespace-only")

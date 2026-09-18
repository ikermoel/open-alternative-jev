"""Typed questions and typed answers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = len(LETTERS)


@dataclass(frozen=True)
class Choice:
    """A question whose answer must be exactly one of `options`.

    The model never writes free text: it only ranks the option letters, so the answer is always
    one of the options you gave. Up to 26 options per question in this version.
    """
    question: str
    options: Sequence[str]
    name: str | None = None

    def __post_init__(self):
        options = tuple(str(o) for o in self.options)
        if not 2 <= len(options) <= MAX_OPTIONS:
            raise ValueError(f"a Choice needs between 2 and {MAX_OPTIONS} options, got {len(options)}")
        if len(set(options)) != len(options):
            raise ValueError("options must be distinct")
        object.__setattr__(self, "options", options)

    @property
    def n(self) -> int:
        return len(self.options)


def yes_no(question: str, name: str | None = None) -> Choice:
    return Choice(question, ("yes", "no"), name)


def scale(question: str, low: int, high: int, name: str | None = None) -> Choice:
    """A rubric score: integer options from `low` to `high` inclusive."""
    if high <= low:
        raise ValueError("high must be greater than low")
    return Choice(question, tuple(str(i) for i in range(low, high + 1)), name)


@dataclass
class Decision:
    """The typed answer to one Choice.

    `probabilities` is the model's distribution over the options, normalized to sum to 1 (after any
    temperature scaling). `confidence` is the probability of the chosen option. Raw confidence from an
    uncalibrated model is usually too high; see `so1.calibration`.
    """
    question: Choice
    probabilities: list[float]
    raw_scores: list[float] = field(default_factory=list, repr=False)

    @property
    def index(self) -> int:
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__)

    @property
    def choice(self) -> str:
        return self.question.options[self.index]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.index]

    def as_dict(self) -> dict:
        return {"name": self.question.name, "question": self.question.question, "choice": self.choice,
                "confidence": self.confidence,
                "probabilities": dict(zip(self.question.options, self.probabilities))}

"""Backend contract: given token sequences and readout positions, return label scores.

A score is a logit or log-probability of each label token at each readout position. Scores only need
to be correct up to an additive constant per position, because the decider applies a softmax over the
labels of each question.
"""
from __future__ import annotations

import math
from typing import Protocol, Sequence


class Backend(Protocol):
    tokenizer: object

    def label_scores(self, sequences: Sequence[Sequence[int]], positions: Sequence[Sequence[int]],
                     label_ids: Sequence[int]) -> list[list[list[float]]]:
        """scores[s][p][l]: score of label l at readout position positions[s][p] of sequences[s]."""
        ...


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]

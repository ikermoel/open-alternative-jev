"""Temperature scaling: one scalar that turns over-confident raw probabilities into calibrated ones.

Fit it on a few hundred labelled decisions that are NOT the ones you evaluate on. In the benchmarks
(Qwen3.6-27B, 8-bit) the fitted temperature was 1.3-1.5 and halved the expected calibration error.
"""
from __future__ import annotations

import math
from typing import Sequence

from .backends.base import softmax
from .schema import Decision


def _scale(probabilities: Sequence[float], temperature: float) -> list[float]:
    return softmax([math.log(max(p, 1e-30)) for p in probabilities], temperature)


def negative_log_likelihood(probabilities: Sequence[Sequence[float]], correct: Sequence[int]) -> float:
    return -sum(math.log(max(p[y], 1e-30)) for p, y in zip(probabilities, correct)) / len(correct)


def expected_calibration_error(probabilities: Sequence[Sequence[float]], correct: Sequence[int], bins: int = 10) -> float:
    n = len(correct)
    conf = [max(p) for p in probabilities]
    hit = [max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probabilities, correct)]
    total = 0.0
    for b in range(bins):
        ix = [i for i, c in enumerate(conf) if min(int(c * bins), bins - 1) == b]
        if ix:
            total += len(ix) / n * abs(sum(conf[i] for i in ix) / len(ix) - sum(hit[i] for i in ix) / len(ix))
    return total


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, probabilities: Sequence[Sequence[float]], correct: Sequence[int],
            grid: Sequence[float] | None = None) -> "TemperatureScaler":
        """Pick the temperature that minimizes negative log-likelihood on labelled data."""
        if len(probabilities) != len(correct) or not correct:
            raise ValueError("probabilities and correct must be non-empty and the same length")
        grid = list(grid) if grid is not None else [x / 20 for x in range(4, 81)]  # 0.2 .. 4.0
        self.temperature = min(grid, key=lambda t: negative_log_likelihood([_scale(p, t) for p in probabilities], correct))
        return self

    def transform(self, probabilities: Sequence[Sequence[float]]) -> list[list[float]]:
        return [_scale(p, self.temperature) for p in probabilities]

    def apply(self, decision: Decision) -> Decision:
        return Decision(decision.question, _scale(decision.probabilities, self.temperature), decision.raw_scores)


def cross_fit_temperature(probabilities: Sequence[Sequence[float]], correct: Sequence[int], folds: int = 2):
    """Scale every example with a temperature fitted on the other folds. Returns (scaled, temperatures)."""
    n = len(correct)
    scaled: list[list[float] | None] = [None] * n
    temperatures = []
    for k in range(folds):
        eval_ix = [i for i in range(n) if i % folds == k]
        fit_ix = [i for i in range(n) if i % folds != k]
        scaler = TemperatureScaler().fit([probabilities[i] for i in fit_ix], [correct[i] for i in fit_ix])
        temperatures.append(scaler.temperature)
        for i in eval_ix:
            scaled[i] = _scale(probabilities[i], scaler.temperature)
    return scaled, temperatures

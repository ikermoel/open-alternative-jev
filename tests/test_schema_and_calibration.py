import math
import random

import pytest

from so1 import Choice, Decision, TemperatureScaler, cross_fit_temperature, expected_calibration_error, scale, yes_no


def test_choice_validation():
    with pytest.raises(ValueError):
        Choice("q", ["only one"])
    with pytest.raises(ValueError):
        Choice("q", ["a", "a"])
    with pytest.raises(ValueError):
        Choice("q", [str(i) for i in range(27)])
    assert yes_no("ok?").options == ("yes", "no")
    assert scale("how good?", 1, 5).options == ("1", "2", "3", "4", "5")


def test_decision_properties():
    d = Decision(Choice("q", ["a", "b", "c"]), [0.2, 0.7, 0.1])
    assert d.index == 1 and d.choice == "b" and d.confidence == 0.7
    assert d.as_dict()["probabilities"] == {"a": 0.2, "b": 0.7, "c": 0.1}


def _overconfident_dataset(n=600, seed=0):
    """Correct 70 % of the time but confident 90 %: the fitted temperature must be > 1."""
    rng = random.Random(seed)
    probs, labels = [], []
    for _ in range(n):
        correct = rng.random() < 0.7
        p = [0.05, 0.05, 0.9]
        rng.shuffle(p)
        top = p.index(0.9)
        labels.append(top if correct else (top + 1) % 3)
        probs.append(p)
    return probs, labels


def test_temperature_scaling_reduces_overconfidence():
    probs, labels = _overconfident_dataset()
    raw = expected_calibration_error(probs, labels)
    scaler = TemperatureScaler().fit(probs, labels)
    scaled = scaler.transform(probs)
    assert scaler.temperature > 1.0
    assert expected_calibration_error(scaled, labels) < raw
    assert all(abs(sum(p) - 1) < 1e-9 for p in scaled)
    # argmax is unchanged by temperature scaling
    assert all(max(range(3), key=a.__getitem__) == max(range(3), key=b.__getitem__) for a, b in zip(probs, scaled))


def test_cross_fit_never_uses_eval_fold():
    probs, labels = _overconfident_dataset()
    scaled, temps = cross_fit_temperature(probs, labels, folds=2)
    assert len(temps) == 2 and all(t > 1 for t in temps)
    assert all(s is not None and abs(sum(s) - 1) < 1e-9 for s in scaled)


def test_apply_to_decision():
    d = Decision(Choice("q", ["a", "b"]), [0.9, 0.1])
    out = TemperatureScaler(2.0).apply(d)
    assert out.choice == "a" and out.confidence < 0.9 and math.isclose(sum(out.probabilities), 1.0)


def test_option_orders():
    from so1.decider import option_orders
    assert option_orders(4, 1) == [[0, 1, 2, 3]]
    assert option_orders(2, 2) == [[0, 1], [1, 0]]
    assert option_orders(2, 5) == [[0, 1], [1, 0]]          # only two distinct orders exist
    assert option_orders(4, 3) == [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0]]
    assert option_orders(3, 4) == [[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]]

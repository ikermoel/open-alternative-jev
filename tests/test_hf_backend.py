import math

from so1 import Choice, Decider, yes_no

STATE = ("Order #4411: customer writes 'The blender arrived with a cracked jar. I want my money back, "
         "and please answer today, I leave for a trip tomorrow.'")
QUESTIONS = [
    Choice("What does the customer want?", ["refund", "replacement", "information"], name="intent"),
    yes_no("Is the message urgent?", name="urgent"),
    Choice("Tone of the message", ["angry", "neutral", "friendly"], name="tone"),
]


def test_packed_and_separate_agree_on_first_question(hf_backend):
    decider = Decider(hf_backend)
    packed = decider.decide(STATE, QUESTIONS, mode="packed")
    separate = decider.decide(STATE, QUESTIONS, mode="separate")
    assert len(packed) == len(separate) == 3
    for d in packed + separate:
        assert math.isclose(sum(d.probabilities), 1.0, abs_tol=1e-6)
        assert len(d.probabilities) == d.question.n
    # identical causal prefix -> identical distribution (float32 on CPU)
    for a, b in zip(packed[0].probabilities, separate[0].probabilities):
        assert abs(a - b) < 1e-4


def test_batching_matches_single(hf_backend):
    decider = Decider(hf_backend)
    items = [(STATE, QUESTIONS), ("The sky is blue today.", [yes_no("Is it raining?")])]
    batched = decider.decide_many(items)
    single = [decider.decide(s, qs) for s, qs in items]
    for group_b, group_s in zip(batched, single):
        for db, ds in zip(group_b, group_s):
            for a, b in zip(db.probabilities, ds.probabilities):
                assert abs(a - b) < 1e-4  # right padding must not change any readout


def test_small_model_gets_the_obvious_one(hf_backend):
    d = Decider(hf_backend).decide(STATE, [QUESTIONS[0]])[0]
    assert d.choice == "refund"


def test_temperature_changes_confidence_not_choice(hf_backend):
    hot = Decider(hf_backend, temperature=3.0).decide(STATE, QUESTIONS)
    cold = Decider(hf_backend, temperature=1.0).decide(STATE, QUESTIONS)
    for h, c in zip(hot, cold):
        assert h.choice == c.choice and h.confidence <= c.confidence + 1e-9


def test_permutations_are_order_invariant(hf_backend):
    """With permutations=2 the result must not depend on the order the caller lists the options in."""
    decider = Decider(hf_backend, permutations=2)
    q_ab = Choice("What does the customer want?", ["refund", "replacement"])
    q_ba = Choice("What does the customer want?", ["replacement", "refund"])
    a = decider.decide(STATE, [q_ab])[0]
    b = decider.decide(STATE, [q_ba])[0]
    # same option, same probability, regardless of listing order
    assert abs(a.probabilities[0] - b.probabilities[1]) < 1e-5
    assert abs(a.probabilities[1] - b.probabilities[0]) < 1e-5
    assert abs(sum(a.probabilities) - 1) < 1e-6
    assert a.choice == "refund"


def test_permutations_one_is_unchanged(hf_backend):
    plain = Decider(hf_backend).decide(STATE, QUESTIONS)
    explicit = Decider(hf_backend, permutations=1).decide(STATE, QUESTIONS)
    for p, e in zip(plain, explicit):
        assert p.probabilities == e.probabilities


def test_permutations_mixed_option_counts_and_separate_mode(hf_backend):
    decider = Decider(hf_backend, permutations=3)
    for mode in ("packed", "separate"):
        ds = decider.decide(STATE, QUESTIONS, mode=mode)
        assert [d.question.n for d in ds] == [3, 2, 3]
        assert all(abs(sum(d.probabilities) - 1) < 1e-6 for d in ds)

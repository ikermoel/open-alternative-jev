from so1 import Choice, PromptBuilder
from so1.prompting import render_turn


def test_render_turn_layout():
    q = Choice("Is it spam?", ["yes", "no"])
    first = render_turn(q, "some email", first=True)
    later = render_turn(q, "some email", first=False)
    assert "Context:\nsome email" in first and "A. yes\nB. no" in first
    assert "some email" not in later and "same context" in later
    assert "Context:" not in render_turn(q, None, first=True)


def test_packed_prompt_structure(tokenizer):
    pb = PromptBuilder(tokenizer)
    qs = [Choice("q1", ["a", "b"]), Choice("q2", ["a", "b", "c"]), Choice("q3", ["x", "y"])]
    packed = pb.packed("shared state", qs)
    assert len(packed.positions) == 3 and packed.positions == sorted(packed.positions)
    assert packed.positions[-1] == len(packed.ids) - 1
    text = tokenizer.decode(packed.ids)
    assert text.count("shared state") == 1, "the state must be written exactly once"
    assert text.count("<|im_start|>user") == 3 and text.count("_<|im_end|>") == 2
    # each readout position is the last token of a generation prompt: what follows it is the placeholder
    for p in packed.positions[:-1]:
        assert tokenizer.decode(packed.ids[p + 1:p + 2]) == "_"
    # the first turn is byte-identical to the separate prompt for the same question
    assert pb.separate("shared state", qs)[0].ids == packed.ids[:packed.positions[0] + 1]


def test_label_ids_single_token(tokenizer):
    pb = PromptBuilder(tokenizer)
    ids = pb.label_ids(4)
    assert len(ids) == 4 and len(set(ids)) == 4
    assert [tokenizer.decode([i]) for i in ids] == ["A", "B", "C", "D"]

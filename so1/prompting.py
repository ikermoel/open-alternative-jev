"""Turn a state plus typed questions into one token sequence with one readout position per question.

Layout (ChatML, the default format):

    <|im_start|>user
    <instruction> + <state> + question 1 + options<|im_end|>
    <|im_start|>assistant
    <think></think>            <- readout 1: the distribution over the next token, restricted to A/B/C/...
    _<|im_end|>                <- fixed placeholder; the model's own answer is never inserted
    <|im_start|>user
    question 2 + options<|im_end|>
    <|im_start|>assistant
                               <- readout 2
    ...

Every question is a normal user turn, so the model sees a format it was trained on. The shared state
is written once, in the first turn. Later questions can attend to earlier questions and to the
placeholders, which is what makes one forward pass possible and is also the source of the measured
interference between questions (see the benchmarks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from .schema import Choice, LETTERS

DEFAULT_INSTRUCTION = "Choose the correct option. Reply with only its letter."
FOLLOW_UP_INSTRUCTION = "Answer the following question about the same context. Reply with only the letter."


@dataclass(frozen=True)
class ChatFormat:
    """Template-specific strings. The default is ChatML as used by Qwen; adapt for other templates."""
    user_turn_start: str = "<|im_start|>user"
    separator: str = "_<|im_end|>\n"
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})


@dataclass
class PackedPrompt:
    ids: list[int]
    positions: list[int]  # index of the token whose next-token distribution answers question i
    questions: list[Choice]

    def __len__(self):
        return len(self.ids)


def render_turn(choice: Choice, state: str | None, first: bool, instruction: str = DEFAULT_INSTRUCTION) -> str:
    lines = []
    if state is None or first:
        lines.append(instruction)
        if state is not None:
            lines += ["", "Context:", state]
    else:
        lines.append(FOLLOW_UP_INSTRUCTION)
    lines += ["", f"Question: {choice.question}"]
    lines += [f"{LETTERS[i]}. {option}" for i, option in enumerate(choice.options)]
    return "\n".join(lines)


class PromptBuilder:
    def __init__(self, tokenizer, fmt: ChatFormat | None = None, instruction: str = DEFAULT_INSTRUCTION):
        self.tokenizer = tokenizer
        self.fmt = fmt or ChatFormat()
        self.instruction = instruction
        self._user_start = self._encode(self.fmt.user_turn_start)
        self._separator = self._encode(self.fmt.separator)
        if not self._user_start or not self._separator:
            raise ValueError("chat format strings must tokenize to at least one token")

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    @lru_cache(maxsize=None)
    def label_ids(self, n: int) -> tuple[int, ...]:
        """Token ids of the option letters A.. for `n` options. Each must be exactly one token."""
        ids = []
        for letter in LETTERS[:n]:
            toks = self._encode(letter)
            if len(toks) != 1:
                raise ValueError(f"label {letter!r} is not a single token for this tokenizer: {toks}")
            ids.append(toks[0])
        return tuple(ids)

    def turn_ids(self, text: str) -> list[int]:
        out = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
            return_dict=False, **self.fmt.chat_template_kwargs)
        if hasattr(out, "keys"):  # BatchEncoding on some transformers versions
            out = out["input_ids"]
        seq = list(out)
        if not all(isinstance(t, int) for t in seq):
            raise TypeError("chat template did not return a flat list of token ids")
        return seq

    def _strip_system(self, seq: list[int]) -> list[int]:
        k = len(self._user_start)
        for i in range(len(seq) - k + 1):
            if seq[i:i + k] == self._user_start:
                return seq[i:]
        raise ValueError("could not find the user turn start in the chat prompt; check ChatFormat.user_turn_start")

    def packed(self, state: str | None, questions: Sequence[Choice]) -> PackedPrompt:
        """All questions in one sequence: state once, one readout position per question."""
        questions = list(questions)
        if not questions:
            raise ValueError("at least one question is required")
        ids: list[int] = []
        positions: list[int] = []
        for i, q in enumerate(questions):
            turn = self.turn_ids(render_turn(q, state, first=(i == 0), instruction=self.instruction))
            if i:
                ids.extend(self._separator)
                turn = self._strip_system(turn)
            ids.extend(turn)
            positions.append(len(ids) - 1)
        return PackedPrompt(ids, positions, questions)

    def separate(self, state: str | None, questions: Sequence[Choice]) -> list[PackedPrompt]:
        """One sequence per question, each with the full state: the no-interference baseline."""
        return [self.packed(state, [q]) for q in questions]

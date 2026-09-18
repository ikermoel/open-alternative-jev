"""The Decider: state + typed questions in, typed decisions out."""
from __future__ import annotations

from typing import Sequence

from .backends import load_backend, softmax
from .prompting import ChatFormat, PromptBuilder
from .schema import Choice, Decision

Item = tuple  # (state: str | None, questions: Sequence[Choice])


class Decider:
    """Answers typed questions about a state by reading next-token distributions, never generating text.

    mode="packed":   all questions of an item in one sequence, one forward pass, state written once.
                     Fastest when questions share a state; later questions can be influenced by
                     earlier ones (measured: 6-9 % of individual answers change, accuracy unchanged).
    mode="separate": one sequence per question, each with the full state. No interference; the
                     backend batches them (vLLM additionally shares the state through its prefix cache).
    """

    def __init__(self, backend, *, prompt_builder: PromptBuilder | None = None, fmt: ChatFormat | None = None,
                 temperature: float = 1.0, mode: str = "packed"):
        self.backend = backend
        self.prompts = prompt_builder or PromptBuilder(backend.tokenizer, fmt)
        self.temperature = temperature
        self.mode = mode

    @classmethod
    def from_pretrained(cls, model_id: str, backend: str = "hf", *, fmt: ChatFormat | None = None,
                        temperature: float = 1.0, mode: str = "packed", **backend_kwargs) -> "Decider":
        return cls(load_backend(backend, model_id, **backend_kwargs), fmt=fmt, temperature=temperature, mode=mode)

    def decide(self, state: str | None, questions: Sequence[Choice], mode: str | None = None) -> list[Decision]:
        return self.decide_many([(state, questions)], mode=mode)[0]

    def decide_many(self, items: Sequence[Item], mode: str | None = None) -> list[list[Decision]]:
        mode = mode or self.mode
        if mode not in ("packed", "separate"):
            raise ValueError("mode must be 'packed' or 'separate'")
        items = [(state, list(qs)) for state, qs in items]
        max_options = max((q.n for _, qs in items for q in qs), default=0)
        if max_options == 0:
            return [[] for _ in items]
        label_ids = list(self.prompts.label_ids(max_options))

        if mode == "packed":
            prompts = [self.prompts.packed(state, qs) for state, qs in items]
            scores = self.backend.label_scores([p.ids for p in prompts], [p.positions for p in prompts], label_ids)
            flat_questions = [p.questions for p in prompts]
        else:
            prompts = [p for state, qs in items for p in self.prompts.separate(state, qs)]
            if hasattr(self.backend, "label_scores_last"):
                rows = self.backend.label_scores_last([p.ids for p in prompts], label_ids)
            else:
                rows = [r[0] for r in self.backend.label_scores([p.ids for p in prompts], [p.positions for p in prompts], label_ids)]
            scores, flat_questions, k = [], [], 0
            for state, qs in items:
                scores.append(rows[k:k + len(qs)])
                flat_questions.append(qs)
                k += len(qs)

        return [[self._decision(q, row) for q, row in zip(qs, rows)] for qs, rows in zip(flat_questions, scores)]

    def _decision(self, question: Choice, row: Sequence[float]) -> Decision:
        raw = list(row[:question.n])
        return Decision(question, softmax(raw, self.temperature), raw)

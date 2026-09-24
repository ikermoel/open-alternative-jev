"""The Decider: state + typed questions in, typed decisions out."""
from __future__ import annotations

from typing import Sequence

from .backends import load_backend, softmax
from .prompting import ChatFormat, PromptBuilder
from .schema import Choice, Decision

Item = tuple  # (state: str | None, questions: Sequence[Choice])


def option_orders(n: int, permutations: int) -> list[list[int]]:
    """Deterministic option orderings: the original, then the reversal, then cyclic shifts.

    permutations=1 -> [identity]; 2 -> [identity, reversed]; k>2 adds shifts by 1, 2, ... until k distinct
    orders exist (or all n! are exhausted for tiny n).
    """
    orders = [list(range(n))]
    if permutations >= 2 and n >= 2:
        orders.append(list(reversed(range(n))))
    shift = 1
    while len(orders) < permutations and shift < n:
        order = [(i + shift) % n for i in range(n)]
        if order not in orders:
            orders.append(order)
        shift += 1
    return orders


class Decider:
    """Answers typed questions about a state by reading next-token distributions, never generating text.

    mode="packed":   all questions of an item in one sequence, one forward pass, state written once.
                     Fastest when questions share a state; later questions can be influenced by
                     earlier ones (measured: 6-9 % of individual answers change, accuracy unchanged).
    mode="separate": one sequence per question, each with the full state. No interference; the
                     backend batches them (vLLM additionally shares the state through its prefix cache).

    permutations:    1 (default) reads each question once, options in the order you gave. 2 also asks with
                     the options reversed and averages the two probability vectors, which cancels the
                     model's preference for a position (A before B) at twice the compute; k > 2 adds
                     cyclic shifts. Probabilities are always reported in your original option order.
    """

    def __init__(self, backend, *, prompt_builder: PromptBuilder | None = None, fmt: ChatFormat | None = None,
                 temperature: float = 1.0, mode: str = "packed", permutations: int = 1):
        self.backend = backend
        self.prompts = prompt_builder or PromptBuilder(backend.tokenizer, fmt)
        self.temperature = temperature
        self.mode = mode
        self.permutations = permutations

    @classmethod
    def from_pretrained(cls, model_id: str, backend: str = "hf", *, fmt: ChatFormat | None = None,
                        temperature: float = 1.0, mode: str = "packed", permutations: int = 1, **backend_kwargs) -> "Decider":
        return cls(load_backend(backend, model_id, **backend_kwargs), fmt=fmt, temperature=temperature, mode=mode,
                   permutations=permutations)

    def decide(self, state: str | None, questions: Sequence[Choice], mode: str | None = None,
               permutations: int | None = None) -> list[Decision]:
        return self.decide_many([(state, questions)], mode=mode, permutations=permutations)[0]

    def decide_many(self, items: Sequence[Item], mode: str | None = None, permutations: int | None = None) -> list[list[Decision]]:
        mode = mode or self.mode
        permutations = permutations or self.permutations
        if mode not in ("packed", "separate"):
            raise ValueError("mode must be 'packed' or 'separate'")
        if permutations < 1:
            raise ValueError("permutations must be >= 1")
        items = [(state, list(qs)) for state, qs in items]
        if not any(qs for _, qs in items):
            return [[] for _ in items]

        # One "view" per (item, ordering index): the same questions with their options permuted.
        # orders[i][j] is the option order used for question j of item i in view p (or None if that
        # question has fewer distinct orders than p).
        views = []  # (item index, [(question, order or None)])
        for i, (state, qs) in enumerate(items):
            per_q = [option_orders(q.n, permutations) for q in qs]
            for p in range(permutations):
                if any(p < len(o) for o in per_q):
                    views.append((i, state, [(q, o[p] if p < len(o) else None) for q, o in zip(qs, per_q)]))

        raw_by_view = self._score_views(views, mode)  # list aligned with views: [raw scores per question]

        # Average probability vectors over orderings, mapped back to the original option order.
        out: list[list[Decision]] = []
        for i, (state, qs) in enumerate(items):
            acc = [[0.0] * q.n for q in qs]
            count = [0] * len(qs)
            first_raw: list[list[float] | None] = [None] * len(qs)
            for (vi, _, view_qs), raws in zip(views, raw_by_view):
                if vi != i:
                    continue
                for j, ((q, order), raw) in enumerate(zip(view_qs, raws)):
                    if order is None:
                        continue
                    probs = softmax(raw[:q.n], self.temperature)
                    for pos, orig in enumerate(order):  # position `pos` in this view shows original option `orig`
                        acc[j][orig] += probs[pos]
                    count[j] += 1
                    if first_raw[j] is None:
                        first_raw[j] = list(raw[:q.n])
            out.append([Decision(q, [v / count[j] for v in acc[j]], first_raw[j] or []) for j, q in enumerate(qs)])
        return out

    def _score_views(self, views, mode):
        """Run the backend over every view; return raw label scores per question, in each view's own order."""
        max_options = max(q.n for _, _, vqs in views for q, _ in vqs)
        label_ids = list(self.prompts.label_ids(max_options))
        permuted_views = []
        for vi, state, vqs in views:
            qs = [Choice(q.question, [q.options[k] for k in order], name=q.name) if order is not None else None for q, order in vqs]
            permuted_views.append((state, qs))

        if mode == "packed":
            prompts = [self.prompts.packed(state, [q for q in qs if q is not None]) for state, qs in permuted_views]
            scores = self.backend.label_scores([p.ids for p in prompts], [p.positions for p in prompts], label_ids)
        else:
            prompts = [p for state, qs in permuted_views for p in self.prompts.separate(state, [q for q in qs if q is not None])]
            if hasattr(self.backend, "label_scores_last"):
                rows = self.backend.label_scores_last([p.ids for p in prompts], label_ids)
            else:
                rows = [r[0] for r in self.backend.label_scores([p.ids for p in prompts], [p.positions for p in prompts], label_ids)]
            scores, k = [], 0
            for state, qs in permuted_views:
                n = sum(q is not None for q in qs)
                scores.append(rows[k:k + n])
                k += n

        # Re-insert None placeholders so each view's list aligns with its questions.
        aligned = []
        for (state, qs), rows in zip(permuted_views, scores):
            it = iter(rows)
            aligned.append([next(it) if q is not None else None for q in qs])
        return aligned

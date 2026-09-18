"""vLLM backend.

Two ways to read label distributions:

- `label_scores` (packed prompts, several readout positions per sequence): uses `prompt_logprobs`,
  which returns the top-k log-probabilities at every prompt position. Labels outside the top-k get a
  floor score one nat below the lowest returned value. With k=20 the option letters are practically
  always inside the top-k after an "answer with the letter" instruction, but this path is
  approximate by construction; `missing_labels` counts how often a label was not returned.
- `label_scores_last` (one question per sequence, readout at the last token): generates one token
  with `allowed_token_ids` restricted to the labels and asks for their log-probabilities. Exact, and
  vLLM's prefix caching shares the state across questions automatically. This is the conventional
  "batching with prefix cache" baseline the benchmarks compare against.
"""
from __future__ import annotations

import inspect
from typing import Sequence


def _engine_arg_supported(name: str) -> bool:
    try:
        from vllm import EngineArgs
        import dataclasses
        return name in {f.name for f in dataclasses.fields(EngineArgs)}
    except Exception:
        return False


class VLLMBackend:
    def __init__(self, llm, tokenizer=None, topk: int = 20):
        self.llm = llm
        self.tokenizer = tokenizer if tokenizer is not None else llm.get_tokenizer()
        self.topk = topk
        self.missing_labels = 0

    @classmethod
    def from_pretrained(cls, model_id: str, *, topk: int = 20, **llm_kwargs) -> "VLLMBackend":
        from vllm import LLM
        llm_kwargs.setdefault("enable_prefix_caching", True)
        # logprobs_mode is an engine argument. "processed_logprobs" makes the returned log-probabilities reflect
        # the allowed_token_ids mask, so label_scores_last gets exactly the label distribution.
        if _engine_arg_supported("logprobs_mode"):
            llm_kwargs.setdefault("logprobs_mode", "processed_logprobs")
        return cls(LLM(model=model_id, **llm_kwargs), topk=topk)

    def _params(self, **kwargs):
        from vllm import SamplingParams
        return SamplingParams(**kwargs)

    def _generate(self, sequences, params):
        return self.llm.generate([{"prompt_token_ids": list(s)} for s in sequences], params, use_tqdm=False)

    def _scores_from(self, dist, label_ids) -> list[float]:
        values = {tid: lp.logprob for tid, lp in dist.items()}
        floor = min(values.values()) - 1.0
        scores = []
        for t in label_ids:
            if t in values:
                scores.append(values[t])
            else:
                self.missing_labels += 1
                scores.append(floor)
        return scores

    def label_scores(self, sequences: Sequence[Sequence[int]], positions: Sequence[Sequence[int]],
                     label_ids: Sequence[int]) -> list[list[list[float]]]:
        params = self._params(max_tokens=1, temperature=0.0, prompt_logprobs=self.topk, logprobs=self.topk)
        outputs = self._generate(sequences, params)
        result = []
        for out, seq, ps in zip(outputs, sequences, positions):
            rows = []
            for p in ps:
                if p + 1 < len(seq):
                    dist = out.prompt_logprobs[p + 1]  # distribution that predicted token p+1
                elif p + 1 == len(seq):
                    dist = out.outputs[0].logprobs[0]  # distribution of the first generated token
                else:
                    raise IndexError(f"readout position {p} beyond sequence of length {len(seq)}")
                rows.append(self._scores_from(dist, label_ids))
            result.append(rows)
        return result

    def label_scores_last(self, sequences: Sequence[Sequence[int]], label_ids: Sequence[int]) -> list[list[float]]:
        """Exact label scores at the last position of each sequence, using constrained generation."""
        params = self._params(max_tokens=1, temperature=0.0, allowed_token_ids=list(label_ids), logprobs=max(len(label_ids), self.topk))
        outputs = self._generate(sequences, params)
        return [self._scores_from(out.outputs[0].logprobs[0], label_ids) for out in outputs]

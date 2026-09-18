"""Hugging Face Transformers backend: one forward pass, logits read only at the readout positions."""
from __future__ import annotations

from typing import Sequence

import torch


class HFBackend:
    def __init__(self, model, tokenizer, batch_size: int = 8):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id
        self._supports_logits_to_keep = True

    @classmethod
    def from_pretrained(cls, model_id: str, *, load_in_8bit: bool = False, load_in_4bit: bool = False,
                        dtype=None, device_map="auto", model_class=None, batch_size: int = 8,
                        tokenizer_kwargs: dict | None = None, **model_kwargs) -> "HFBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, **(tokenizer_kwargs or {}))
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        classes = [model_class] if model_class is not None else [AutoModelForCausalLM]
        if model_class is None:
            try:
                from transformers import AutoModelForImageTextToText
                classes.append(AutoModelForImageTextToText)  # multimodal checkpoints such as Qwen3.5
            except ImportError:
                pass
        errors = []
        for klass in classes:
            try:
                model = klass.from_pretrained(model_id, **model_kwargs)
                return cls(model, tokenizer, batch_size=batch_size)
            except (ValueError, KeyError, OSError) as e:
                errors.append(f"{klass.__name__}: {e}")
        raise ValueError("could not load model with any auto class:\n" + "\n".join(errors))

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def label_scores(self, sequences: Sequence[Sequence[int]], positions: Sequence[Sequence[int]],
                     label_ids: Sequence[int]) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        label_index = torch.tensor(list(label_ids), device=self.device)
        for start in range(0, len(sequences), self.batch_size):
            seqs = [list(s) for s in sequences[start:start + self.batch_size]]
            poss = [list(p) for p in positions[start:start + self.batch_size]]
            maxlen = max(len(s) for s in seqs)
            ids = torch.tensor([s + [self.pad_id] * (maxlen - len(s)) for s in seqs], device=self.device)
            mask = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs], device=self.device)
            selected = sorted({p for ps in poss for p in ps})
            logits = self._forward(ids, mask, selected)  # [batch, len(selected), vocab]
            index = {p: i for i, p in enumerate(selected)}
            for b, ps in enumerate(poss):
                rows = logits[b, [index[p] for p in ps]][:, label_index].float().cpu().tolist()
                out.append(rows)
        return out

    def _forward(self, ids, mask, selected):
        if self._supports_logits_to_keep:
            try:
                keep = torch.tensor(selected, device=self.device)
                return self.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
            except TypeError:
                self._supports_logits_to_keep = False
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return logits[:, selected]

"""Three typed decisions about one support ticket in a single forward pass.

    python examples/quickstart.py                       # Qwen2.5-0.5B-Instruct on CPU/GPU, quick
    SO1_MODEL=Qwen/Qwen3.6-27B SO1_8BIT=1 python examples/quickstart.py
    SO1_BACKEND=vllm SO1_MODEL=Qwen/Qwen3.5-4B python examples/quickstart.py
"""
import os
import time

from so1 import Choice, Decider, yes_no

model = os.environ.get("SO1_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
backend = os.environ.get("SO1_BACKEND", "hf")
kwargs = {"load_in_8bit": True} if backend == "hf" and os.environ.get("SO1_8BIT") else {}

decider = Decider.from_pretrained(model, backend=backend, **kwargs)

state = ("Ticket #8813 from a Pro-plan customer: 'Since yesterday's release the export button does nothing. "
         "We have a board meeting Monday and need the PDF. Please fix or tell me a workaround.'")
questions = [
    Choice("Ticket category", ["bug", "feature request", "billing", "question"], name="category"),
    Choice("Urgency", ["low", "medium", "high"], name="urgency"),
    yes_no("Should this be escalated to an engineer?", name="escalate"),
    Choice("Customer sentiment", ["angry", "worried", "neutral", "happy"], name="sentiment"),
]

t0 = time.perf_counter()
decisions = decider.decide(state, questions)
dt = time.perf_counter() - t0
for d in decisions:
    print(f"{d.question.name:10s} -> {d.choice:16s} confidence {d.confidence:.2f}   {d.as_dict()['probabilities']}")
print(f"{len(decisions)} decisions in one forward pass, {dt * 1000:.0f} ms (includes tokenization)")

"""Runs only where vLLM and a GPU are available (e.g. the cluster). Compares vLLM against the HF backend."""
import os

import pytest

from so1 import Choice, Decider, yes_no

vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("vLLM tests need a GPU", allow_module_level=True)

MODEL = os.environ.get("SO1_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
STATE = "Ticket: 'App crashes every time I open the camera tab since the last update. Pixel 8.'"
QUESTIONS = [
    Choice("Category", ["bug", "feature request", "question", "billing"]),
    yes_no("Does the user mention a device?"),
    Choice("Severity", ["low", "medium", "high"]),
]


@pytest.fixture(scope="module")
def vllm_backend():
    import json
    from so1.backends.vllm import VLLMBackend
    extra = json.loads(os.environ.get("SO1_VLLM_KWARGS", "{}"))  # e.g. {"gdn_prefill_backend": "triton"}
    return VLLMBackend.from_pretrained(MODEL, gpu_memory_utilization=float(os.environ.get("SO1_GPU_UTIL", "0.5")),
                                       max_model_len=4096, enforce_eager=True, **extra)


def test_packed_positions_and_shapes(vllm_backend):
    decider = Decider(vllm_backend)
    decisions = decider.decide(STATE, QUESTIONS, mode="packed")
    assert [d.question.n for d in decisions] == [4, 2, 3]
    assert all(abs(sum(d.probabilities) - 1) < 1e-6 for d in decisions)
    assert vllm_backend.missing_labels == 0


def test_separate_uses_exact_path(vllm_backend):
    decider = Decider(vllm_backend)
    decisions = decider.decide(STATE, QUESTIONS, mode="separate")
    assert len(decisions) == 3 and decisions[0].choice == "bug"


def test_packed_first_question_matches_separate(vllm_backend):
    decider = Decider(vllm_backend)
    packed = decider.decide(STATE, QUESTIONS, mode="packed")[0]
    separate = decider.decide(STATE, QUESTIONS, mode="separate")[0]
    # same prefix; bf16 kernels and different code paths allow small numerical differences
    assert max(abs(a - b) for a, b in zip(packed.probabilities, separate.probabilities)) < 0.05


def test_vllm_agrees_with_hf(vllm_backend, hf_backend):
    a = Decider(vllm_backend).decide(STATE, QUESTIONS, mode="separate")
    b = Decider(hf_backend).decide(STATE, QUESTIONS, mode="separate")
    assert [d.choice for d in a] == [d.choice for d in b]

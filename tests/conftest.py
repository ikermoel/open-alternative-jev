import os

import pytest

MODEL = os.environ.get("SO1_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="session")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # offline, or model not available
        pytest.skip(f"tokenizer {MODEL} not available: {e}")


@pytest.fixture(scope="session")
def hf_backend():
    pytest.importorskip("torch")
    from so1.backends.hf import HFBackend
    try:
        return HFBackend.from_pretrained(MODEL, dtype="float32", device_map=None, batch_size=4)
    except Exception as e:
        pytest.skip(f"model {MODEL} not available: {e}")

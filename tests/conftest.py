import os

import pytest

MODEL = os.environ.get("SO1_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# Optional: pin the Hub revision as well as the name, so a re-upload of the
# repo cannot quietly change what the suite is testing. CI sets it.
REVISION = os.environ.get("SO1_TEST_REVISION") or None
# Both fixtures below skip when the model can't be loaded, which is right on a
# laptop and wrong in CI: a green run that skipped everything proves nothing.
# When this is set, a load failure is a failure.
REQUIRE_MODEL = os.environ.get("SO1_TEST_REQUIRE_MODEL") == "1"


def _unavailable(what, exc):
    if REQUIRE_MODEL:
        raise RuntimeError(f"{what} not available and SO1_TEST_REQUIRE_MODEL=1: {exc}") from exc
    pytest.skip(f"{what} not available: {exc}")


@pytest.fixture(scope="session")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    except Exception as e:  # offline, or model not available
        _unavailable(f"tokenizer {MODEL}", e)


@pytest.fixture(scope="session")
def hf_backend():
    pytest.importorskip("torch")
    from so1.backends.hf import HFBackend
    try:
        return HFBackend.from_pretrained(MODEL, dtype="float32", device_map=None, batch_size=4,
                                         revision=REVISION, tokenizer_kwargs={"revision": REVISION})
    except Exception as e:
        _unavailable(f"model {MODEL}", e)

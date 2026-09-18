from .base import Backend, softmax

__all__ = ["Backend", "softmax", "load_backend"]


def load_backend(name: str, model_id: str, **kwargs) -> Backend:
    """Instantiate a backend by name: "hf" (Hugging Face Transformers) or "vllm"."""
    if name == "hf":
        from .hf import HFBackend
        return HFBackend.from_pretrained(model_id, **kwargs)
    if name == "vllm":
        from .vllm import VLLMBackend
        return VLLMBackend.from_pretrained(model_id, **kwargs)
    raise ValueError(f"unknown backend {name!r}; use 'hf' or 'vllm'")

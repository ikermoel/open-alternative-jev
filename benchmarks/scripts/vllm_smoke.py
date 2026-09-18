"""Minimal vLLM engine start plus one generate with prompt_logprobs. Must be a real file: vLLM spawns
a worker process that re-imports the main module.

    SO1_TEST_MODEL=/path python benchmarks/scripts/vllm_smoke.py
"""
import os
import time

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    t = time.time()
    import json
    extra = json.loads(os.environ.get("SO1_VLLM_KWARGS", "{}"))
    llm = LLM(model=os.environ["SO1_TEST_MODEL"], gpu_memory_utilization=float(os.environ.get("SO1_GPU_UTIL", "0.45")),
              max_model_len=4096, enforce_eager=True, enable_prefix_caching=True, **extra)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=3, temperature=0, logprobs=3, prompt_logprobs=3))
    print("VLLM_SMOKE_OK", repr(out[0].outputs[0].text), "prompt_logprobs_len", len(out[0].prompt_logprobs),
          "load_s", round(time.time() - t, 1), flush=True)

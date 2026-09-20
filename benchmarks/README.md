# Benchmarks

Everything the top-level README reports comes from here. Nothing is trained; no inference engine is patched.

| Path | What |
|---|---|
| `scripts/prepare_v2.py` | Samples 1200 MMLU test questions and 250 RACE-H test passages (4 questions each) at pinned dataset revisions. Writes `data/`. |
| `scripts/benchmark_v2.py` | Paired scoring of the same questions in modes A (one at a time), A_pad (shape-noise control), B{k} (padded batch), C{k} (packed), C{k}_rot (rotated order). Writes predictions, timings and a summary per run. |
| `scripts/analyze_run.py` | Post-hoc: regresses forward time on tokens, separates numerical noise from interference by position, cross-fitted temperature scaling. |
| `scripts/library_compare.py` | Runs the `so1` library itself (HF and vLLM backends, packed and separate) on RACE-H and compares them. |
| `scripts/make_figures.py` | Regenerates `figures/` from `results/`. |
| `scripts/benchmark.py`, `prepare.py`, `report.py` | The original 300-question pilot. Kept for the record; its result was corrected by `analyze_run.py` (see `docs/RESULTS.md`). |
| `results/` | Per-question predictions, per-call timings, summaries and analyses of every run. |
| `docs/RESULTS.md` | Full write-up: the question, the pilot, the padding correction, the corrected runs, interference, the 4B finding and calibration. `docs/RESULTS.es.md` is the Spanish original. |

Model: Qwen3.6-27B (official post-trained checkpoint, revision pinned in `data/manifest.json`), bitsandbytes
LLM.int8 with BF16 non-quantized modules, `sdpa` attention, Transformers 5.16, one NVIDIA H200 MIG 2g.35gb
slice. Prompts are zero-shot with thinking disabled; probabilities are a softmax over the option letters only.

`data/race1000.jsonl` is not committed (RACE is distributed for research use); rebuild it with `prepare_v2.py`.
The `.sbatch` files are the Slurm scripts used on the cluster and contain site-specific paths.

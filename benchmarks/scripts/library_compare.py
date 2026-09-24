"""End-to-end check of the so1 library on RACE-H: HF and vLLM backends, packed and separate modes.

Runs the same passages through every (backend, mode) pair, reports accuracy, agreement with the HF
separate baseline, wall-clock time per question (including tokenization) and tokens sent. The vLLM
"separate" mode is the conventional baseline: one request per question, state shared through the
prefix cache.

    python benchmarks/scripts/library_compare.py --model /path/or/id --passages 100 --backends hf,vllm --out results/lib_x
"""
import argparse, json, statistics, time
from pathlib import Path

from so1 import Choice, Decider, expected_calibration_error
from so1.prompting import PromptBuilder


def load_items(path, passages):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    items = []
    for g in sorted(groups)[:passages]:
        rs = groups[g]
        items.append((rs[0]["context"], [Choice(r["question"], r["choices"], name=r["id"]) for r in rs], [r["answer"] for r in rs]))
    return items


def run(decider, items, mode, chunk):
    t0 = time.perf_counter()
    decisions = []
    for i in range(0, len(items), chunk):
        decisions.extend(decider.decide_many([(s, qs) for s, qs, _ in items[i:i + chunk]], mode=mode))
    elapsed = time.perf_counter() - t0
    return decisions, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/race1000.jsonl"))
    ap.add_argument("--passages", type=int, default=100)
    ap.add_argument("--backends", default="hf,vllm")
    ap.add_argument("--chunk", type=int, default=8, help="items per decide_many call")
    ap.add_argument("--hf-8bit", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--permutations", type=int, default=1, help="option orderings to average (1 = as given)")
    ap.add_argument("--option-order", choices=["original", "reversed"], default="original")
    ap.add_argument("--modes", default="separate,packed")
    ap.add_argument("--vllm-kwargs", default="{}", help='JSON of extra vllm.LLM kwargs, e.g. {"gdn_prefill_backend": "triton"}')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    vllm_kwargs = json.loads(args.vllm_kwargs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(args.data, args.passages)
    if args.option_order == "reversed":  # reverse every question's options and its answer index
        items = [(s, [Choice(q.question, list(reversed(q.options)), name=q.name) for q in qs], [q.n - 1 - a for q, a in zip(qs, ans)])
                 for s, qs, ans in items]
    n_questions = sum(len(qs) for _, qs, _ in items)
    print(f"ITEMS {len(items)} passages, {n_questions} questions", flush=True)

    results = {}
    baseline = None
    for backend in args.backends.split(","):
        if backend == "hf":
            import torch
            decider = Decider.from_pretrained(args.model, backend="hf", load_in_8bit=args.hf_8bit, dtype=torch.bfloat16,
                                              device_map={"": 0}, batch_size=8)
        else:
            # max_num_seqs: hybrid (Mamba/GDN) models need one cache block per decode sequence; vLLM's default of
            # 1024 exceeds what a 35 GB slice holds. 256 is plenty for this workload.
            decider = Decider.from_pretrained(args.model, backend="vllm", gpu_memory_utilization=args.gpu_util,
                                              max_model_len=8192, max_num_seqs=args.max_num_seqs,
                                              enable_prefix_caching=True, **vllm_kwargs)
        pb = PromptBuilder(decider.backend.tokenizer)
        tokens = {"packed": sum(len(pb.packed(s, qs)) for s, qs, _ in items),
                  "separate": sum(len(p) for s, qs, _ in items for p in pb.separate(s, qs))}
        for mode in args.modes.split(","):
            run(decider, items[:2], mode, args.chunk)  # warm up kernels / prefix cache
            decider.permutations = args.permutations
            decisions, elapsed = run(decider, items, mode, args.chunk)
            flat = [d for group in decisions for d in group]
            answers = [a for _, _, ans in items for a in ans]
            probs = [d.probabilities for d in flat]
            preds = [d.index for d in flat]
            key = f"{backend}/{mode}"
            r = dict(backend=backend, mode=mode, permutations=args.permutations, option_order=args.option_order, questions=len(flat),
                     accuracy=statistics.mean(p == a for p, a in zip(preds, answers)),
                     ece=expected_calibration_error(probs, answers),
                     mean_confidence=statistics.mean(max(p) for p in probs),
                     seconds=elapsed, questions_per_second=len(flat) / elapsed, tokens=tokens[mode],
                     missing_labels=getattr(decider.backend, "missing_labels", 0))
            if baseline is None:
                baseline = (key, preds, probs)
            else:
                r["agreement_with_" + baseline[0].replace("/", "_")] = statistics.mean(p == q for p, q in zip(preds, baseline[1]))
                r["mean_abs_prob_diff_vs_baseline"] = statistics.mean(abs(x - y) for p, q in zip(probs, baseline[2]) for x, y in zip(p, q))
            results[key] = r
            print("RESULT", json.dumps(r), flush=True)
            (out / f"{backend}_{mode}.jsonl").write_text("\n".join(json.dumps(d.as_dict() | {"answer": a}) for d, a in zip(flat, answers)) + "\n")
        del decider
        try:
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
    (out / "summary.json").write_text(json.dumps(dict(model=args.model, passages=len(items), results=results), indent=2))
    print("SUMMARY", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()

"""LocalLLaMA/typed-decisions with so1 (any open model, zero-shot) or with Laya, scored the same way.

The benchmark: 400 test cases, each one shared state plus 5 typed questions (`noul` yes/no, `choice`,
ordered `score`), gold = mean of three teacher samples (soft distributions). Metrics follow the third-party
scorer in Luni/laya-jev-benchmark (bench/eval.py), so numbers are comparable to the published Laya 0.766
and Jev 0.727 rows: accuracy on the argmax, Brier/KL/TV against the full gold distribution, ECE from the
predicted confidence, MAE and within-1 for score questions.

    # so1, zero-shot, all five questions in one forward pass, temperature fitted on 200 train cases
    python benchmarks/scripts/typed_decisions.py --backend so1 --model Qwen/Qwen3.6-27B --load-8bit \
        --mode packed --calibrate 200 --out benchmarks/results/td_qwen27b_packed
    # Laya, base checkpoint or the one fine-tuned on this benchmark
    python benchmarks/scripts/typed_decisions.py --backend laya --model /path/to/convaiinnovations/laya \
        --out benchmarks/results/td_laya_base
"""
import argparse, json, math, statistics, time
from pathlib import Path

import numpy as np

DATASET = "LocalLLaMA/typed-decisions"


def load_split(split, data_dir=None):
    import pyarrow.parquet as pq
    if data_dir:
        path = Path(data_dir) / f"{split}-00000-of-00001.parquet"
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET, f"all/{split}-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    for r in rows:
        for k in ("state", "questions", "gold"):
            r[k] = json.loads(r[k])
    return rows


# ----------------------------------------------------------------------------- so1 rendering

def render_state(state) -> str:
    return json.dumps(state, indent=1, ensure_ascii=False) if not isinstance(state, str) else state


def question_to_choice(qid, qdef):
    """Map one typed question to a so1 Choice. Option descriptions stay in the prompt: the benchmark says
    stripping them makes a different, easier task."""
    from so1 import Choice
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice":
        keys = list(crit.keys())
        options = [f"{k}: {crit[k]}" for k in keys]
        return Choice(qdef["instructions"], options, name=qid), keys
    if t == "noul":
        keys = ["true", "false"]
        desc = crit if isinstance(crit, dict) else {}
        options = [f"true: {desc.get('true', 'the statement is true')}", f"false: {desc.get('false', 'the statement is false')}"]
        return Choice("Is the following statement true? " + qdef["instructions"], options, name=qid), keys
    if t == "score":
        levels = list(crit) if isinstance(crit, list) else [crit[k] for k in sorted(crit, key=int)]
        keys = [str(i) for i in range(len(levels))]
        options = [f"{i}: {d}" for i, d in enumerate(levels)]
        return Choice(qdef["instructions"] + " Pick the level that fits best.", options, name=qid), keys
    raise ValueError(f"unknown question type {t}")


def decision_to_answer(qdef, keys, probs):
    t = qdef["type"]
    p = {k: float(v) for k, v in zip(keys, probs)}
    if t == "choice":
        best = max(p, key=p.get)
        return {"choice": best, "probabilities": p, "confidence": p[best]}
    if t == "noul":
        return {"noul": p["true"], "probabilities": p, "confidence": abs(2 * p["true"] - 1)}
    expected = sum(int(k) * v for k, v in p.items())
    return {"score": expected, "probabilities": p, "confidence": max(p.values())}


class So1Runner:
    def __init__(self, model, mode, load_8bit, device, dtype, temperature=1.0, engine="hf", gpu_util=0.8,
                 permutations=1, option_order="original"):
        self.permutations, self.option_order = permutations, option_order
        import torch
        from so1 import Decider
        if engine == "vllm":
            self.decider = Decider.from_pretrained(model, backend="vllm", temperature=temperature, mode=mode,
                                                   gpu_memory_utilization=gpu_util, max_model_len=8192, max_num_seqs=64,
                                                   enable_prefix_caching=True)
        else:
            kw = {}
            if load_8bit:
                kw["load_in_8bit"] = True
            else:
                kw["dtype"] = getattr(torch, dtype)
                kw["device_map"] = device if device != "cpu" else None
            self.decider = Decider.from_pretrained(model, backend="hf", temperature=temperature, mode=mode, **kw)
        self.mode = mode

    def predict(self, state, questions):
        from so1 import Choice
        pairs = [question_to_choice(qid, qdef) for qid, qdef in questions.items()]
        if self.option_order == "reversed":  # present every question's options in reverse; keys follow
            pairs = [(Choice(c.question, list(reversed(c.options)), name=c.name), list(reversed(keys))) for c, keys in pairs]
        decisions = self.decider.decide(render_state(state), [c for c, _ in pairs], mode=self.mode, permutations=self.permutations)
        return {qid: decision_to_answer(qdef, keys, d.probabilities)
                for (qid, qdef), (_, keys), d in zip(questions.items(), pairs, decisions)}


class LayaRunner:
    def __init__(self, model, device):
        import laya
        self.agent = laya.Agent(model, device=device)

    def predict(self, state, questions):
        return self.agent.predict(state, questions)["answers"]


# ----------------------------------------------------------------------------- scoring (port of Luni's eval.py)

def ece_score(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def score(records):
    acc, soft, brier, kl, tv, mae, w1, confs, corrects = ([] for _ in range(9))
    for rec in records:
        questions, gold, pred = rec["questions"], rec["gold"], rec["answers"]
        for qid, qdef in questions.items():
            p, g, t = pred[qid], gold[qid], qdef["type"]
            if t == "choice":
                keys = list(qdef["criteria"].keys())
                is_corr = float(p["choice"] == str(g["label"]))
                pp = np.array([p["probabilities"].get(k, 1e-6) for k in keys])
                gp = np.array([g["probabilities"].get(k, 1e-6) for k in keys])
            elif t == "noul":
                pv = p["noul"]
                gv = g.get("noul", g.get("probabilities", {}).get("true", 0.5))
                is_corr = float(("true" if pv >= 0.5 else "false") == str(g["label"]).lower())
                pp, gp = np.array([1 - pv, pv]), np.array([1 - gv, gv])
            else:
                n = len(qdef.get("criteria", []))
                pp = np.array([p["probabilities"].get(str(i), 0.0) for i in range(n)])
                mae.append(abs(p["score"] - g.get("score", 0.0)))
                w1.append(float(abs(p["score"] - g.get("score", 0.0)) <= 1.0))
                if pp.sum() > 0:
                    pp = pp / pp.sum()
                    p_lvl, conf = int(np.argmax(pp)), float(pp.max())
                else:
                    p_lvl, conf = int(round(p["score"])), 0.5
                is_corr = float(p_lvl == int(g.get("label", round(g.get("score", 0.0)))))
                acc.append(is_corr); corrects.append(is_corr); confs.append(conf)
                continue
            pp = pp / pp.sum(); gp = gp / gp.sum()
            acc.append(is_corr); corrects.append(is_corr); confs.append(float(pp.max()))
            soft.append(float((pp * gp).sum()))
            brier.append(float(((pp - gp) ** 2).sum()))
            tv.append(float(0.5 * np.abs(pp - gp).sum()))
            kl.append(float((gp * np.log(np.clip(gp / pp, 1e-12, 1e4))).sum()))
    return dict(n=len(acc), acc=float(np.mean(acc)), soft=float(np.mean(soft)) if soft else None,
                brier=float(np.mean(brier)) if brier else None, kl=float(np.mean(kl)) if kl else None,
                tv=float(np.mean(tv)) if tv else None, ece=ece_score(confs, corrects),
                mae=float(np.mean(mae)) if mae else 0.0, within1=float(np.mean(w1)) if w1 else 0.0)


def fit_temperature(runner, train_rows):
    """Temperature that minimizes cross-entropy against the soft gold on train cases (never test)."""
    from so1.backends.base import softmax
    samples = []
    for r in train_rows:
        pairs = [question_to_choice(qid, qdef) for qid, qdef in r["questions"].items()]
        decisions = runner.decider.decide(render_state(r["state"]), [c for c, _ in pairs], mode=runner.mode)
        for (qid, qdef), (_, keys), d in zip(r["questions"].items(), pairs, decisions):
            g = r["gold"][qid]
            gp = [g["probabilities"].get(k, 0.0) for k in keys] if qdef["type"] != "noul" else [g["probabilities"].get("true", 0.5), g["probabilities"].get("false", 0.5)]
            samples.append((d.raw_scores, gp))
    grid = [x / 20 for x in range(4, 101)]  # 0.2 .. 5.0

    def ce(T):
        total = 0.0
        for raw, gp in samples:
            pp = softmax(raw, T)
            total -= sum(g * math.log(max(p, 1e-12)) for g, p in zip(gp, pp))
        return total / len(samples)
    return min(grid, key=ce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["so1", "laya"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["packed", "separate"], default="packed")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf", help="so1 backend engine")
    ap.add_argument("--gpu-util", type=float, default=0.8, help="vLLM gpu_memory_utilization")
    ap.add_argument("--permutations", type=int, default=1, help="so1: option orderings to average (1 = as given, 2 = + reversed)")
    ap.add_argument("--option-order", choices=["original", "reversed"], default="original", help="so1: present options as given or reversed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--calibrate", type=int, default=0, help="so1: fit a temperature on this many train cases (0 = raw)")
    ap.add_argument("--limit", type=int, default=0, help="test cases to run (0 = all 400)")
    ap.add_argument("--data-dir", default=None, help="directory with train/test parquet files (offline nodes)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    test = load_split("test", args.data_dir)
    if args.limit:
        test = test[:args.limit]
    print(f"CASES {len(test)} test | backend {args.backend} | model {args.model} | mode {args.mode}", flush=True)

    temperature = 1.0
    if args.backend == "so1":
        runner = So1Runner(args.model, args.mode, args.load_8bit, args.device, args.dtype, engine=args.engine, gpu_util=args.gpu_util,
                           permutations=args.permutations, option_order=args.option_order)
        if args.calibrate:
            train = load_split("train", args.data_dir)
            per_wf = args.calibrate // 4
            train = [r for wf in sorted({r["workflow"] for r in train}) for r in [x for x in train if x["workflow"] == wf][:per_wf]]
            temperature = fit_temperature(runner, train)
            runner.decider.temperature = temperature
            print(f"TEMPERATURE {temperature} fitted on {len(train)} train cases", flush=True)
    else:
        runner = LayaRunner(args.model, args.device)

    runner.predict(test[0]["state"], test[0]["questions"])  # warm-up
    records, latencies = [], []
    with (out / "predictions.jsonl").open("w") as f:
        for i, r in enumerate(test):
            t0 = time.perf_counter()
            answers = runner.predict(r["state"], r["questions"])
            latencies.append((time.perf_counter() - t0) * 1000)
            rec = dict(id=r["id"], workflow=r["workflow"], questions=r["questions"], gold=r["gold"], answers=answers)
            records.append(rec)
            f.write(json.dumps(dict(id=r["id"], workflow=r["workflow"], answers=answers)) + "\n")
            if (i + 1) % 50 == 0:
                print(f"PROGRESS {i + 1}/{len(test)}", flush=True)

    summary = dict(backend=args.backend, engine=args.engine if args.backend == "so1" else None, model=args.model, mode=args.mode if args.backend == "so1" else None,
                   permutations=args.permutations, option_order=args.option_order,
                   load_8bit=args.load_8bit, temperature=temperature, cases=len(records),
                   ms_per_case_p50=statistics.median(latencies), ms_per_case_mean=statistics.mean(latencies),
                   overall=score(records),
                   by_workflow={wf: score([r for r in records if r["workflow"] == wf]) for wf in sorted({r["workflow"] for r in records})},
                   by_type={t: score([dict(r, questions={q: d for q, d in r["questions"].items() if d["type"] == t},
                                           answers={q: a for q, a in r["answers"].items() if r["questions"][q]["type"] == t})
                                      for r in records]) for t in ["choice", "noul", "score"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    o = summary["overall"]
    print(f"RESULT acc={o['acc']:.4f} brier={o['brier']:.4f} ece={o['ece']:.4f} kl={o['kl']:.4f} mae={o['mae']:.3f} "
          f"p50={summary['ms_per_case_p50']:.1f}ms T={temperature} perm={args.permutations} order={args.option_order} | "
          + json.dumps({k: round(v['acc'], 3) for k, v in summary['by_workflow'].items()})
          + " | by_type " + json.dumps({k: round(v['acc'], 3) for k, v in summary['by_type'].items()}), flush=True)


if __name__ == "__main__":
    main()

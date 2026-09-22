"""RACE-H with Laya: one passage as the state, its four questions as `choice` questions in one call.

Options are addressed as A-D with the option text as the description, which is Laya's native choice format.
Records the share of passages longer than the checkpoint's context (Laya base: 512 tokens; the
typed-decisions checkpoint: 1024), because those get truncated by the model, not by us.

    python benchmarks/scripts/race_laya.py --model /path/to/convaiinnovations/laya --out benchmarks/results/race_laya_base
"""
import argparse, json, statistics, time
from pathlib import Path

LETTERS = "ABCD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/race1000.jsonl"))
    ap.add_argument("--passages", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import laya
    agent = laya.Agent(args.model, device=args.device)
    max_len = agent.cfg.get("max_len", 512)
    tok = agent.tok

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    items = [groups[g] for g in sorted(groups)[:args.passages]]
    print(f"PASSAGES {len(items)} | model {args.model} | max_len {max_len}", flush=True)

    def questions_for(qs):
        return {r["id"]: {"type": "choice", "instructions": r["question"],
                          "criteria": {LETTERS[i]: opt for i, opt in enumerate(r["choices"])}} for r in qs}

    agent.predict(items[0][0]["context"], questions_for(items[0]))  # warm-up
    records, latencies, truncated = [], [], 0
    with (out / "predictions.jsonl").open("w") as f:
        for qs in items:
            passage = qs[0]["context"]
            if len(tok(passage)["input_ids"]) > max_len:
                truncated += 1
            t0 = time.perf_counter()
            answers = agent.predict(passage, questions_for(qs))["answers"]
            latencies.append((time.perf_counter() - t0) * 1000)
            for r in qs:
                a = answers[r["id"]]
                pred = LETTERS.index(a["choice"])
                probs = [a["probabilities"].get(L, 0.0) for L in LETTERS[:len(r["choices"])]]
                rec = dict(id=r["id"], group=r["group"], answer=r["answer"], prediction=pred, probabilities=probs, confidence=a.get("confidence"))
                records.append(rec)
                f.write(json.dumps(rec) + "\n")

    correct = [r["prediction"] == r["answer"] for r in records]
    conf = [max(r["probabilities"]) for r in records]
    n = len(records)
    ece = 0.0
    for b in range(10):
        ix = [i for i, c in enumerate(conf) if min(int(c * 10), 9) == b]
        if ix:
            ece += len(ix) / n * abs(statistics.mean(conf[i] for i in ix) - statistics.mean(correct[i] for i in ix))
    summary = dict(model=args.model, passages=len(items), questions=n, max_len=max_len,
                   passages_over_context=truncated, accuracy=statistics.mean(correct), ece_10_bins=ece,
                   mean_confidence=statistics.mean(conf), ms_per_passage_p50=statistics.median(latencies),
                   questions_per_second=n / (sum(latencies) / 1000))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("RESULT", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

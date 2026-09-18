"""Post-hoc analysis of a finished run. No model dependencies, no GPU.

Answers three questions the summary does not:
1. Is the throughput difference between modes explained by token counts (padding)?
2. How many answer flips are numerical noise (identical causal prefix) versus interference?
3. Does temperature scaling, fitted without touching the evaluated half, improve calibration?
"""
import argparse, json, math, statistics
from collections import defaultdict
from pathlib import Path


def fit_line(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:  # constant token count (A_pad): no slope to estimate
        return my, float('nan'), float('nan')
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope, statistics.correlation(xs, ys)


def ece(probs, labels, bins=10):
    n = len(probs)
    conf = [max(p) for p in probs]
    correct = [max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probs, labels)]
    total = 0.0
    for b in range(bins):
        ix = [i for i, c in enumerate(conf) if min(int(c * bins), bins - 1) == b]
        if ix:
            total += len(ix) / n * abs(statistics.mean(conf[i] for i in ix) - statistics.mean(correct[i] for i in ix))
    return total


def scale(probs, T):
    out = []
    for p in probs:
        logits = [math.log(max(x, 1e-30)) / T for x in p]
        m = max(logits)
        z = sum(math.exp(l - m) for l in logits)
        out.append([math.exp(l - m) / z for l in logits])
    return out


def nll(probs, labels):
    return statistics.mean(-math.log(max(p[y], 1e-30)) for p, y in zip(probs, labels))


def fit_temperature(probs, labels):
    grid = [x / 20 for x in range(4, 81)]  # 0.2 .. 4.0
    return min(grid, key=lambda T: nll(scale(probs, T), labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    args = ap.parse_args()
    root = Path(args.run)
    rows = [json.loads(l) for l in (root / 'predictions.jsonl').read_text().splitlines() if l.strip()]
    calls = [json.loads(l) for l in (root / 'timings.jsonl').read_text().splitlines() if l.strip()]
    modes = sorted({r['mode'] for r in rows}, key=lambda m: (m != 'A', m))
    report = {}

    # 1. Time versus tokens. Same slope across modes means same cost per processed token.
    timing = {}
    for mode in modes:
        cs = [c for c in calls if c['mode'] == mode]
        if len(cs) < 3:
            continue
        intercept, slope, r = fit_line([c['padded_tokens'] for c in cs], [c['seconds'] for c in cs])
        real = sum(c['input_tokens'] for c in cs)
        padded = sum(c['padded_tokens'] for c in cs)
        timing[mode] = dict(calls=len(cs), fixed_seconds_per_call=intercept, ms_per_padded_token=1000 * slope,
                            pearson_r=r, padding_waste=1 - real / padded, seconds=sum(c['seconds'] for c in cs),
                            real_tokens=real, padded_tokens=padded)
    report['time_vs_tokens'] = timing

    # 2. Flips against A by position. Position 0 of a concatenated sequence shares A's causal prefix exactly,
    #    so its flips are the numerical noise floor for this checkpoint and kernel set.
    A = {r['id']: r for r in rows if r['mode'] == 'A'}
    flips = {}
    for mode in modes:
        if mode == 'A':
            continue
        by_pos = defaultdict(list)
        for r in rows:
            if r['mode'] == mode and r['id'] in A:
                by_pos[r['position']].append(r)
        flips[mode] = {}
        for pos, rs in sorted(by_pos.items()):
            a = [A[r['id']] for r in rs]
            flips[mode][str(pos)] = dict(
                n=len(rs),
                accuracy=statistics.mean(r['prediction'] == r['answer'] for r in rs),
                flips_vs_A=sum(r['prediction'] != x['prediction'] for r, x in zip(rs, a)),
                mean_abs_prob_diff=statistics.mean(abs(p - q) for r, x in zip(rs, a) for p, q in zip(r['probabilities'], x['probabilities'])),
                max_abs_prob_diff=max(abs(p - q) for r, x in zip(rs, a) for p, q in zip(r['probabilities'], x['probabilities'])))
    report['flips_by_position'] = flips

    # 3. Two-fold cross-fitted temperature scaling per mode. Never evaluates on the fold used for fitting.
    calib = {}
    for mode in modes:
        rs = sorted((r for r in rows if r['mode'] == mode), key=lambda r: r['id'])
        probs = [r['probabilities'] for r in rs]
        labels = [r['answer'] for r in rs]
        folds = [list(range(0, len(rs), 2)), list(range(1, len(rs), 2))]
        scaled = [None] * len(rs)
        temps = []
        for k in range(2):
            fit_ix, eval_ix = folds[1 - k], folds[k]
            T = fit_temperature([probs[i] for i in fit_ix], [labels[i] for i in fit_ix])
            temps.append(T)
            for i, p in zip(eval_ix, scale([probs[i] for i in eval_ix], T)):
                scaled[i] = p
        calib[mode] = dict(n=len(rs), temperatures=temps,
                           ece_raw=ece(probs, labels), ece_scaled=ece(scaled, labels),
                           nll_raw=nll(probs, labels), nll_scaled=nll(scaled, labels),
                           mean_confidence_raw=statistics.mean(max(p) for p in probs),
                           mean_confidence_scaled=statistics.mean(max(p) for p in scaled),
                           accuracy=statistics.mean(r['prediction'] == r['answer'] for r in rs))
    report['temperature_scaling_2fold'] = calib

    (root / 'analysis.json').write_text(json.dumps(report, indent=2))
    print('TIME VS TOKENS')
    for m, t in timing.items():
        print(f"  {m:10s} calls={t['calls']:4d} fixed={t['fixed_seconds_per_call']:.3f}s  {t['ms_per_padded_token']:.2f} ms/token  r={t['pearson_r']:.3f}  padding_waste={t['padding_waste']:.1%}")
    print('FLIPS VS A BY POSITION')
    for m, d in flips.items():
        for pos, v in d.items():
            print(f"  {m:10s} pos={pos} n={v['n']:4d} acc={v['accuracy']:.3f} flips={v['flips_vs_A']:3d} mean|dp|={v['mean_abs_prob_diff']:.4f} max|dp|={v['max_abs_prob_diff']:.3f}")
    print('TEMPERATURE SCALING (2-fold cross-fitted)')
    for m, c in calib.items():
        print(f"  {m:10s} T={c['temperatures']}  ECE {c['ece_raw']:.4f} -> {c['ece_scaled']:.4f}  NLL {c['nll_raw']:.4f} -> {c['nll_scaled']:.4f}  conf {c['mean_confidence_raw']:.3f} -> {c['mean_confidence_scaled']:.3f}  acc={c['accuracy']:.3f}")


if __name__ == '__main__':
    main()

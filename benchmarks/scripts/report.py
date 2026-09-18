"""Summarize a completed or partial run without model dependencies."""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('run'); args=p.parse_args(); root=Path(args.run)
rows=[json.loads(l) for l in (root/'predictions.jsonl').read_text().splitlines() if l.strip()]
calls=[json.loads(l) for l in (root/'timings.jsonl').read_text().splitlines() if l.strip()]
modes=defaultdict(list)
for row in rows: modes[row['mode']].append(row)
lines=['# MMLU pilot results','',f'Run: `{root.name}`. '+('Complete.' if (root/'summary.json').exists() else '**Partial; do not draw conclusions yet.**'),'', '| Mode | Questions | Accuracy | Questions / GPU forward second | Peak allocated GiB |','|---|---:|---:|---:|---:|']
for mode in ['A','B','C','C_rotated']:
    rs=modes[mode]; cs=[c for c in calls if c['mode']==mode]
    if not rs: continue
    seconds=sum(c['seconds'] for c in cs)
    lines.append(f'| {mode} | {len(rs)} | {sum(r["prediction"]==r["answer"] for r in rs)/len(rs):.2%} | {len(rs)/seconds:.3f} | {max(c["peak_allocated_gib"] for c in cs):.2f} |')
a={r['id']:r for r in modes['A']}
first=[r for r in modes['C'] if r['position']==0 and r['id'] in a]
if first:
    delta=max(abs(p-q) for r in first for p,q in zip(r['probabilities'],a[r['id']]['probabilities']))
    lines.extend(['',f'First-position C vs A maximum probability difference: {delta:.6f}. These share identical causal prefixes; nonzero differences can reflect kernels/quantization.'])
lines.extend(['','This measures forced A/B/C/D decisions, with no generated reasoning. Probabilities are normalized over these four labels and are not calibrated. Timing excludes load, tokenization and tensor preparation. C uses concatenated chat turns and fixed placeholders. All modes use the same quantized checkpoint. These are pilot results, not a reproduction of published MMLU scores.'])
text='\n'.join(lines)+'\n'; (root/'REPORT.md').write_text(text); print(text)

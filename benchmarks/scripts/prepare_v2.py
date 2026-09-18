"""Network-only data preparation for the v2 runs. No model execution.

Writes two datasets in one common row format:
  id, group, context (may be null), question, choices (4 strings), answer (0-3), subject

- data/mmlu1200.jsonl: the 300 pilot questions first (same ids), then 900 new disjoint questions.
  Groups of 12 are assigned by position, so every group of 3, 6 or 12 is a prefix-aligned slice.
- data/race1000.jsonl: 250 RACE-high test passages that have exactly 4 questions; group = passage.
Revisions of both datasets are pinned in data/manifest_v2.json.
"""
import json, random
from collections import defaultdict
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

root = Path(__file__).resolve().parents[1]
api = HfApi()
manifest = json.loads((root / 'data/manifest.json').read_text())
SEED = 20260918

# MMLU: reuse the pilot revision so the 300 pilot ids stay valid.
mmlu_rev = manifest['dataset_revision']
path = hf_hub_download('cais/mmlu', 'all/test-00000-of-00001.parquet', repo_type='dataset', revision=mmlu_rev)
rows = pq.read_table(path).to_pylist()
pilot = [json.loads(l) for l in (root / 'data/mmlu300.jsonl').read_text().splitlines()]
pilot_ids = [r['id'] for r in pilot]
assert all(rows[i]['question'] == r['question'] for i, r in zip(pilot_ids, pilot)), 'pilot ids do not match this revision'
remaining = [i for i in range(len(rows)) if i not in set(pilot_ids)]
new_ids = random.Random(SEED).sample(remaining, 900)
with (root / 'data/mmlu1200.jsonl').open('w') as f:
    for k, i in enumerate(pilot_ids + new_ids):
        r = rows[i]
        f.write(json.dumps(dict(id=f'mmlu{i}', group=k // 12, context=None, question=r['question'],
                                choices=list(r['choices']), answer=int(r['answer']), subject=r['subject'])) + '\n')

# RACE high, test split: one passage, several 4-option questions. Keep passages with exactly 4 questions.
race_rev = api.dataset_info('ehovy/race').sha
path = hf_hub_download('ehovy/race', 'high/test-00000-of-00001.parquet', repo_type='dataset', revision=race_rev)
race = pq.read_table(path).to_pylist()
by_passage = defaultdict(list)
for k, r in enumerate(race):
    by_passage[r['article']].append((k, r))
eligible = sorted((art for art, qs in by_passage.items() if len(qs) == 4), key=lambda a: by_passage[a][0][0])
chosen = random.Random(SEED).sample(eligible, 250)
with (root / 'data/race1000.jsonl').open('w') as f:
    for g, art in enumerate(chosen):
        for k, r in by_passage[art]:
            assert len(r['options']) == 4 and r['answer'] in 'ABCD'
            f.write(json.dumps(dict(id=f'race{k}', group=g, context=r['article'], question=r['question'],
                                    choices=list(r['options']), answer='ABCD'.index(r['answer']), subject='race_high')) + '\n')

manifest_v2 = dict(seed=SEED,
                   mmlu=dict(dataset='cais/mmlu', revision=mmlu_rev, split='test', count=1200, pilot_prefix=300, source_count=len(rows)),
                   race=dict(dataset='ehovy/race', config='high', revision=race_rev, split='test', passages=250, questions=1000,
                             eligible_passages=len(eligible), source_questions=len(race)))
(root / 'data/manifest_v2.json').write_text(json.dumps(manifest_v2, indent=2))
print('DATA_V2_READY', json.dumps(manifest_v2), flush=True)

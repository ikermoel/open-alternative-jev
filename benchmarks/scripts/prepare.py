"""Network-only preparation; no model execution on the login node."""
import json, random
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import pyarrow.parquet as pq
root = Path(__file__).resolve().parents[1]
api = HfApi()
model = 'Qwen/Qwen3.6-27B'
revision = api.model_info(model).sha
ds_revision = api.dataset_info('cais/mmlu').sha
path = hf_hub_download('cais/mmlu', 'all/test-00000-of-00001.parquet', repo_type='dataset', revision=ds_revision)
rows = pq.read_table(path).to_pylist()
indices = random.Random(20260917).sample(range(len(rows)), 300)
with (root / 'data/mmlu300.jsonl').open('w') as f:
    for i in indices:
        r = rows[i]
        f.write(json.dumps(dict(id=i, **r)) + '\n')
manifest = dict(model=model, revision=revision, dataset='cais/mmlu', dataset_revision=ds_revision, split='test', seed=20260917, count=300, source_count=len(rows))
(root / 'data/manifest.json').write_text(json.dumps(manifest, indent=2))
print('DATA_READY', manifest, flush=True)
path = snapshot_download(model, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'], max_workers=4)
print('MODEL_READY', path, flush=True)
(root / 'data/model_path.txt').write_text(path)

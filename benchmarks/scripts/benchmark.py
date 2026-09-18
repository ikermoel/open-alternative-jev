"""Paired zero-shot MMLU label scoring. No generation, training, or engine edits."""
import argparse, hashlib, json, math, random, statistics, time
from pathlib import Path

LETTERS = 'ABCD'

def question_text(row):
    return 'Choose the correct answer. Reply with only A, B, C, or D.\n\n' + row['question'] + '\n' + '\n'.join(f'{c}. {s}' for c, s in zip(LETTERS, row['choices']))

def pack(sequences, separator):
    ids, positions = [], []
    for i, seq in enumerate(sequences):
        if i:
            ids.extend(separator)
        ids.extend(seq)
        positions.append(len(ids) - 1)
    return ids, positions

def metrics(rows):
    n = len(rows)
    correct = [r['prediction'] == r['answer'] for r in rows]
    conf = [max(r['probabilities']) for r in rows]
    ece = 0.
    for b in range(10):
        ix = [i for i, c in enumerate(conf) if min(int(c * 10), 9) == b]
        if ix:
            ece += len(ix)/n * abs(statistics.mean(conf[i] for i in ix) - statistics.mean(correct[i] for i in ix))
    return dict(n=n, accuracy=statistics.mean(correct), ece_10_bins=ece,
                nll=statistics.mean(-math.log(max(r['probabilities'][r['answer']], 1e-30)) for r in rows),
                brier=statistics.mean(sum((p-int(j == r['answer']))**2 for j,p in enumerate(r['probabilities'])) for r in rows),
                accuracy_by_position={str(pos):statistics.mean(r['prediction']==r['answer'] for r in rows if r['position']==pos) for pos in sorted({r['position'] for r in rows})})

def summarize(records, calls):
    by_mode = {m: [r for r in records if r['mode']==m] for m in ['A','B','C','C_rotated']}
    summary = {m: metrics(rs) for m, rs in by_mode.items()}
    for m in summary:
        cs=[c for c in calls if c['mode']==m]
        seconds=sum(c['seconds'] for c in cs)
        summary[m].update(forward_seconds=seconds, questions_per_second=len(by_mode[m])/seconds,
                          peak_allocated_gib=max(c['peak_allocated_gib'] for c in cs),
                          peak_reserved_gib=max(c['peak_reserved_gib'] for c in cs))
    for other in ['B', 'C', 'C_rotated']:
        a={r['id']:r for r in by_mode['A']}; b={r['id']:r for r in by_mode[other]}
        ids=sorted(a)
        # Resample complete triplets: questions sharing C context are not independent.
        groups={}
        for i in ids:
            groups.setdefault(a[i]['group'], []).append(int(b[i]['prediction']==b[i]['answer'])-int(a[i]['prediction']==a[i]['answer']))
        group_values=list(groups.values()); rng=random.Random(42); boot=[]
        for _ in range(2000):
            sample=[v for g in rng.choices(group_values,k=len(group_values)) for v in g]
            boot.append(statistics.mean(sample))
        boot.sort()
        summary[other]['vs_A']=dict(accuracy_delta=statistics.mean(v for g in group_values for v in g),
            cluster_bootstrap_95_ci=[boot[50],boot[1949]], answer_disagreement=statistics.mean(a[i]['prediction']!=b[i]['prediction'] for i in ids),
            max_probability_difference=max(abs(x-y) for i in ids for x,y in zip(a[i]['probabilities'],b[i]['probabilities'])))
    c={r['id']:r for r in by_mode['C']}; d={r['id']:r for r in by_mode['C_rotated']}
    summary['order_disagreement']=statistics.mean(c[i]['prediction']!=d[i]['prediction'] for i in c)
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--count',type=int,default=300); ap.add_argument('--out',required=True); ap.add_argument('--cpu-smoke',action='store_true'); args=ap.parse_args()
    if args.cpu_smoke:
        ids, pos=pack([[11,12],[21,22,23],[31]], [99,98])
        assert [ids[p] for p in pos]==[12,23,31] and pos==[1,6,9]
        print('CPU_SMOKE_OK: causal readout indices and separator placement'); return
    import torch, transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    root=Path(__file__).resolve().parents[1]; out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in (root/'data/mmlu300.jsonl').read_text().splitlines()][:args.count]
    assert len(rows)%3==0
    model_path=(root/'data/model_path.txt').read_text().strip()
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    label_ids=[tok.encode(c,add_special_tokens=False) for c in LETTERS]
    assert all(len(x)==1 for x in label_ids), label_ids
    label_ids=[x[0] for x in label_ids]
    prompts={r['id']:tok.apply_chat_template([{'role':'user','content':question_text(r)}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False) for r in rows}
    assert all(isinstance(seq, list) and all(isinstance(t, int) for t in seq) for seq in prompts.values())
    separator=tok.encode('_<|im_end|>\n',add_special_tokens=False)
    meta=dict(manifest=json.loads((root/'data/manifest.json').read_text()), torch=torch.__version__,transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
              quantization='bitsandbytes LLM.int8; BF16 nonquantized modules', labels=label_ids, separator_ids=separator,
              sample_prompt=tok.decode(prompts[rows[0]['id']]),count=len(rows),
              protocol='zero-shot forced label scoring, thinking disabled; C concatenates complete chat prompts with fixed underscore assistant placeholders; no generated answers reused',
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('LOADING',meta,flush=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,quantization_config=BitsAndBytesConfig(load_in_8bit=True),dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    print('MODEL_LOADED',model.get_memory_footprint(),flush=True)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.inference_mode()
    def score(group,mode):
        seqs=[prompts[r['id']] for r in group]
        if mode.startswith('C'):
            joined,positions=pack(seqs,separator); inputs=[joined]; targets=[(0,p) for p in positions]
        else:
            inputs=seqs; targets=[(i,len(seq)-1) for i,seq in enumerate(seqs)]
        maxlen=max(map(len,inputs))
        ids=torch.tensor([s+[pad]*(maxlen-len(s)) for s in inputs],device='cuda')
        mask=torch.tensor([[1]*len(s)+[0]*(maxlen-len(s)) for s in inputs],device='cuda')
        selected=sorted({p for _,p in targets}); indexes={p:i for i,p in enumerate(selected)}
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
        outputs=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=torch.tensor(selected,device='cuda'))
        logits=torch.stack([outputs.logits[b,indexes[p],label_ids].float() for b,p in targets])
        probabilities=logits.softmax(-1).cpu().tolist()
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started
        call=dict(mode=mode,seconds=elapsed,questions=len(group),input_tokens=sum(map(len,inputs)),padded_tokens=ids.numel(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        return probabilities,call

    # Warm all shapes before timed observations; cold model load excluded explicitly.
    for mode in ['A','B','C']:
        score(rows[:1] if mode=='A' else rows[:3],mode)
    print('WARMUP_OK',flush=True)
    records=[]; calls=[]
    with (out/'predictions.jsonl').open('w') as predfile, (out/'timings.jsonl').open('w') as timefile:
        for gi,start in enumerate(range(0,len(rows),3)):
            group=rows[start:start+3]; modes=['A','B','C','C_rotated']; random.Random(gi).shuffle(modes)
            for mode in modes:
                ordered=group[1:]+group[:1] if mode=='C_rotated' else group
                chunks=[[r] for r in ordered] if mode=='A' else [ordered]
                for chunk in chunks:
                    probs,call=score(chunk,mode); call['group']=gi; calls.append(call); timefile.write(json.dumps(call)+'\n'); timefile.flush()
                    for position,(r,p) in enumerate(zip(chunk,probs)):
                        rec=dict(id=r['id'],subject=r['subject'],group=gi,mode=mode,position=position,answer=r['answer'],prediction=max(range(4),key=lambda j:p[j]),probabilities=p)
                        records.append(rec); predfile.write(json.dumps(rec)+'\n'); predfile.flush()
            print('PROGRESS',gi+1,len(rows)//3,flush=True)
    summary=summarize(records,calls); (out/'summary.json').write_text(json.dumps(summary,indent=2)); print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__': main()

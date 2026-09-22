"""Paired zero-shot label scoring, v2. No generation, training, or engine edits.

Row format (data/*.jsonl): id, group, context|null, question, choices[4], answer(0-3), subject.
Rows sharing `group` form one super-group of size G (12 for MMLU, 4 for RACE). Modes:

  A        one question per forward (reference)
  A_pad    same as A, right-padded to a fixed length: isolates shape/kernel noise from interference
  B{k}     k independent sequences in one padded batch (conventional batching)
  C{k}     k chat prompts concatenated in one sequence, fixed `_` assistant placeholders between them;
           the next-token distribution is read at each answer boundary
  C{k}_rot same as C{k} with the questions rotated by one (order sensitivity)

With a shared context (RACE), A/B put the passage in every question's turn, while C states the passage
only in the first turn; later turns carry only the question. That is the shared-state use case.
"""
import argparse, hashlib, json, math, random, statistics, time
from pathlib import Path

LETTERS = 'ABCD'
INSTRUCTION = 'Choose the correct answer. Reply with only A, B, C, or D.'


def options_block(row):
    return row['question'] + '\n' + '\n'.join(f'{c}. {s}' for c, s in zip(LETTERS, row['choices']))


def user_turn(row, with_context):
    if row.get('context') is None:
        return INSTRUCTION + '\n\n' + options_block(row)
    if with_context:
        return ('Read the passage and answer the question. Reply with only A, B, C, or D.\n\nPassage:\n'
                + row['context'] + '\n\nQuestion: ' + options_block(row))
    return 'Answer the following question about the same passage. Reply with only A, B, C, or D.\n\nQuestion: ' + options_block(row)


def pack(sequences, separator):
    """Concatenate token lists with a separator; return ids and the readout index of each sequence's last token."""
    ids, positions = [], []
    for i, seq in enumerate(sequences):
        if i:
            ids.extend(separator)
        ids.extend(seq)
        positions.append(len(ids) - 1)
    return ids, positions


def parse_modes(spec):
    modes = []
    for m in spec.split(','):
        m = m.strip()
        if m in ('A', 'A_pad'):
            modes.append((m, 1))
        elif m[0] in 'BC':
            k = int(m[1:].split('_')[0])
            modes.append((m, k))
        else:
            raise ValueError(m)
    return modes


def metrics(rows):
    n = len(rows)
    correct = [r['prediction'] == r['answer'] for r in rows]
    conf = [max(r['probabilities']) for r in rows]
    ece = 0.
    for b in range(10):
        ix = [i for i, c in enumerate(conf) if min(int(c * 10), 9) == b]
        if ix:
            ece += len(ix) / n * abs(statistics.mean(conf[i] for i in ix) - statistics.mean(correct[i] for i in ix))
    positions = sorted({r['position'] for r in rows})
    return dict(n=n, accuracy=statistics.mean(correct), ece_10_bins=ece,
                nll=statistics.mean(-math.log(max(r['probabilities'][r['answer']], 1e-30)) for r in rows),
                brier=statistics.mean(sum((p - int(j == r['answer'])) ** 2 for j, p in enumerate(r['probabilities'])) for r in rows),
                accuracy_by_position={str(p): statistics.mean(r['prediction'] == r['answer'] for r in rows if r['position'] == p) for p in positions})


def cluster_bootstrap(values_by_group, resamples=2000, seed=42):
    groups = list(values_by_group.values())
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        sample = [v for g in rng.choices(groups, k=len(groups)) for v in g]
        boot.append(statistics.mean(sample))
    boot.sort()
    return [boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]]


def compare(reference, other):
    ref = {r['id']: r for r in reference}
    oth = {r['id']: r for r in other}
    ids = sorted(set(ref) & set(oth))
    by_group = {}
    for i in ids:
        by_group.setdefault(ref[i]['group'], []).append(int(oth[i]['prediction'] == oth[i]['answer']) - int(ref[i]['prediction'] == ref[i]['answer']))
    flips_by_position = {}
    for i in ids:
        p = str(oth[i]['position'])
        flips_by_position.setdefault(p, [0, 0])
        flips_by_position[p][0] += oth[i]['prediction'] != ref[i]['prediction']
        flips_by_position[p][1] += 1
    return dict(n=len(ids),
                accuracy_delta=statistics.mean(v for g in by_group.values() for v in g),
                cluster_bootstrap_95_ci=cluster_bootstrap(by_group),
                answer_disagreement=statistics.mean(oth[i]['prediction'] != ref[i]['prediction'] for i in ids),
                flips_by_position={p: f'{a}/{b}' for p, (a, b) in sorted(flips_by_position.items())},
                mean_abs_probability_difference=statistics.mean(abs(x - y) for i in ids for x, y in zip(ref[i]['probabilities'], oth[i]['probabilities'])),
                max_probability_difference=max(abs(x - y) for i in ids for x, y in zip(ref[i]['probabilities'], oth[i]['probabilities'])))


def summarize(records, calls, modes):
    by_mode = {m: [r for r in records if r['mode'] == m] for m, _ in modes}
    summary = {}
    for m, rs in by_mode.items():
        if not rs:
            continue
        summary[m] = metrics(rs)
        cs = [c for c in calls if c['mode'] == m]
        seconds = sum(c['seconds'] for c in cs)
        summary[m].update(calls=len(cs), forward_seconds=seconds, questions_per_second=len(rs) / seconds,
                          real_tokens=sum(c['input_tokens'] for c in cs), padded_tokens=sum(c['padded_tokens'] for c in cs),
                          peak_allocated_gib=max(c['peak_allocated_gib'] for c in cs), peak_reserved_gib=max(c['peak_reserved_gib'] for c in cs))
    for m in summary:
        if m != 'A' and by_mode.get('A'):
            summary[m]['vs_A'] = compare(by_mode['A'], by_mode[m])
        if m.startswith('C') and m.endswith('_rot'):
            base = m[:-4]
            if by_mode.get(base):
                summary[m]['vs_' + base] = compare(by_mode[base], by_mode[m])
        if m.startswith('C') and not m.endswith('_rot'):
            b = 'B' + m[1:]
            if by_mode.get(b):
                summary[m]['vs_' + b] = compare(by_mode[b], by_mode[m])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='jsonl in data/')
    ap.add_argument('--group-size', type=int, required=True, help='rows per super-group (12 mmlu, 4 race)')
    ap.add_argument('--modes', required=True, help='e.g. A,A_pad,B3,C3,C3_rot,C6,C12')
    ap.add_argument('--count', type=int, default=None, help='rows to use (multiple of group size)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cpu-smoke', action='store_true', help='index packing check only, no model')
    ap.add_argument('--model', default=None, help='override data/model_path.txt (local smoke tests)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-quant', action='store_true', help='load without bitsandbytes (local smoke tests)')
    args = ap.parse_args()

    if args.cpu_smoke:
        ids, pos = pack([[11, 12], [21, 22, 23], [31]], [99, 98])
        assert [ids[p] for p in pos] == [12, 23, 31] and pos == [1, 6, 9]
        assert parse_modes('A,A_pad,B3,C12,C6_rot') == [('A', 1), ('A_pad', 1), ('B3', 3), ('C12', 12), ('C6_rot', 6)]
        print('CPU_SMOKE_OK: causal readout indices, separator placement, mode parsing')
        return

    import torch, transformers
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(l) for l in (root / 'data' / args.data).read_text().splitlines() if l.strip()]
    if args.count:
        rows = rows[:args.count]
    G = args.group_size
    assert len(rows) % G == 0, (len(rows), G)
    supergroups = [rows[i:i + G] for i in range(0, len(rows), G)]
    assert all(len({r['group'] for r in sg}) == 1 for sg in supergroups), 'super-groups must align with the group field'
    modes = parse_modes(args.modes)
    assert all(G % k == 0 for _, k in modes), 'every k must divide the group size'
    use_cuda = args.device == 'cuda'

    model_path = args.model or (root / 'data/model_path.txt').read_text().strip()
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    label_ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(x) == 1 for x in label_ids), label_ids
    label_ids = [x[0] for x in label_ids]

    def chat_ids(text):
        seq = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, return_dict=False,
                                      add_generation_prompt=True, enable_thinking=False)
        assert isinstance(seq, list) and all(isinstance(t, int) for t in seq)
        return seq

    # Two token sequences per row: the full turn (used by A, B and first C position) and the follow-up turn
    # (used by later C positions). A follow-up turn drops the shared context, if any, and any system block the
    # chat template emits (Qwen3.6 emits none; Qwen2.5 emits one), so C is one continuous multi-turn chat.
    user_start = tok.encode('<|im_start|>user', add_special_tokens=False)

    def strip_system(seq):
        for i in range(len(seq) - len(user_start) + 1):
            if seq[i:i + len(user_start)] == user_start:
                return seq[i:]
        raise ValueError('no user turn found in chat prompt')

    full = {r['id']: chat_ids(user_turn(r, True)) for r in rows}
    follow = {r['id']: strip_system(chat_ids(user_turn(r, False))) for r in rows}
    system_block_stripped = any(len(follow[r['id']]) != len(chat_ids(user_turn(r, False))) for r in rows[:1])
    separator = tok.encode('_<|im_end|>\n', add_special_tokens=False)
    pad_len = -(-max(map(len, full.values())) // 64) * 64
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    meta = dict(data=args.data, manifest_v2=json.loads((root / 'data/manifest_v2.json').read_text()), model_path=model_path,
                torch=torch.__version__, transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0) if use_cuda else args.device,
                gpu_bytes=torch.cuda.get_device_properties(0).total_memory if use_cuda else None,
                quantization='none' if args.no_quant else 'bitsandbytes LLM.int8; BF16 nonquantized modules',
                labels=label_ids, separator_ids=separator, pad_len=pad_len, group_size=G, modes=args.modes, count=len(rows),
                system_block_stripped_in_follow_up_turns=system_block_stripped,
                sample_full_prompt=tok.decode(full[rows[0]['id']]), sample_follow_prompt=tok.decode(follow[rows[0]['id']]),
                protocol='zero-shot forced label scoring, thinking disabled; C concatenates chat turns with fixed underscore '
                         'assistant placeholders; shared context only in the first C turn; no generated answers reused',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('LOADING', json.dumps({k: v for k, v in meta.items() if 'prompt' not in k}), flush=True)

    if args.no_quant:
        from transformers import AutoModelForCausalLM
        dtype = torch.bfloat16 if use_cuda else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=dtype)
        except (ValueError, KeyError):  # multimodal checkpoints such as Qwen3.5
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True, dtype=dtype)
        model = model.to(args.device).eval()
    else:
        from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
    print('MODEL_LOADED', model.get_memory_footprint(), flush=True)

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    @torch.inference_mode()
    def score(group, mode):
        """Return per-question probabilities over A/B/C/D and a timing record for one forward."""
        if mode.startswith('C'):
            seqs = [full[group[0]['id']]] + [follow[r['id']] for r in group[1:]]
            joined, positions = pack(seqs, separator)
            inputs, targets = [joined], [(0, p) for p in positions]
        else:
            inputs = [full[r['id']] for r in group]
            targets = [(i, len(s) - 1) for i, s in enumerate(inputs)]
        maxlen = pad_len if mode == 'A_pad' else max(map(len, inputs))
        assert maxlen >= max(map(len, inputs))
        ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in inputs], device=args.device)
        mask = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in inputs], device=args.device)
        selected = sorted({p for _, p in targets})
        index = {p: i for i, p in enumerate(selected)}
        sync()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=torch.tensor(selected, device=args.device))
        logits = torch.stack([outputs.logits[b, index[p], label_ids].float() for b, p in targets])
        probabilities = logits.softmax(-1).cpu().tolist()
        sync()
        elapsed = time.perf_counter() - started
        call = dict(mode=mode, seconds=elapsed, questions=len(group), input_tokens=sum(map(len, inputs)), padded_tokens=ids.numel(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if use_cuda else 0.,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30 if use_cuda else 0.)
        return probabilities, call

    # Warm every mode's kernels before timed observations.
    for m, k in modes:
        score(supergroups[0][:k], m)
    print('WARMUP_OK', flush=True)

    records, calls = [], []
    with (out / 'predictions.jsonl').open('w') as predfile, (out / 'timings.jsonl').open('w') as timefile:
        for gi, sg in enumerate(supergroups):
            order = list(modes)
            random.Random(gi).shuffle(order)
            for m, k in order:
                for si in range(0, G, k):
                    chunk = sg[si:si + k]
                    if m.endswith('_rot'):
                        chunk = chunk[1:] + chunk[:1]
                    probs, call = score(chunk, m)
                    call.update(group=gi, subgroup=si // k)
                    calls.append(call)
                    timefile.write(json.dumps(call) + '\n')
                    for position, (r, p) in enumerate(zip(chunk, probs)):
                        rec = dict(id=r['id'], subject=r['subject'], group=gi, subgroup=si // k, mode=m, k=k, position=position,
                                   answer=r['answer'], prediction=max(range(4), key=lambda j: p[j]), probabilities=p)
                        records.append(rec)
                        predfile.write(json.dumps(rec) + '\n')
                predfile.flush()
                timefile.flush()
            print('PROGRESS', gi + 1, len(supergroups), flush=True)
    summary = summarize(records, calls, modes)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

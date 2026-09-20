# What happens when you ask a model N questions in one forward pass

A write-up of the experiments behind Open Alternative to Jev, in the order they happened, including the
result we got wrong and how we found out. Every number comes from a file in `benchmarks/results/`; the
scripts that produced them are in `benchmarks/scripts/`. Spanish original: `RESULTS.es.md`.

## 1. The question

A causal language model produces a next-token distribution at every position of its input in a single
forward pass. Chat APIs hide this: you get one generated answer per call. But if you write a state (a
document, a ticket, a log line) followed by several questions, each ending in an answer slot, you can read
the distribution at every slot at once and restrict it to a fixed set of option letters. No text is
generated, every answer is one of your options, and it comes with a probability.

That is the mechanism TypeSafe's Jev product exposes. We wanted to know three things about doing it with a
stock open-weights model, no training:

- Does reading several answers from one sequence keep the accuracy of asking one question at a time?
- Is it actually cheaper, and compared with what?
- Are the probabilities worth anything?

Throughout, three modes are compared on the same questions with the same model:

- **A**: one question per forward pass. The reference.
- **B**: k independent sequences in one padded batch. Conventional batching.
- **C**: k questions concatenated into one sequence, one readout position per question. The idea under test.

Setup for the 27B runs: `Qwen/Qwen3.6-27B`, the official post-trained checkpoint at a pinned revision,
bitsandbytes LLM.int8 with BF16 non-quantized modules, Transformers 5.16, one NVIDIA H200 MIG 2g.35gb slice
(35 GB). Zero-shot, thinking disabled, the chat template's own turns. Each question is rendered as a user
turn; in C the turns are separated by a fixed `_` placeholder answer, never by the model's own prediction.
Probabilities are a softmax over the option letters only. Timing is GPU forward time, excluding model load,
tokenization and tensor preparation, after a warm-up of every shape.

Two facts about this model matter later. Its 64 layers are 48 gated-DeltaNet linear-attention layers and
16 full-attention layers; on the cluster the linear layers ran on PyTorch fallback kernels (no `fla`,
`triton` or `flash-attn`), so absolute throughput is slow and only ratios between modes are meaningful. And
because the linear layers carry a recurrent state, a packed sequence cannot be split into independent
segments with an attention mask, so B has to be a real padded batch.

## 2. The first result: 1.4x faster than batching

Pilot on 300 uniformly sampled MMLU test questions, groups of three (`results/mmlu_32677666/`):

| Mode | Accuracy | Questions / s |
|---|---:|---:|
| A: one at a time | 84.3 % | 3.14 |
| B: batch of 3 | 84.0 % | 4.01 |
| C: 3 packed | 84.0 % | 5.57 |
| C, rotated order | 84.3 % | 5.57 |

Accuracy held, and C processed 1.39x more questions per second than batching. That looked like the
headline: "packing beats batching".

## 3. Why it was wrong

The pilot logged, for every forward call, its wall time, the number of real tokens and the number of tokens
after padding. Regressing time on padded tokens per mode (`scripts/analyze_run.py`):

| Mode | Fixed cost per call | Cost per padded token | r | Tokens wasted on padding |
|---|---:|---:|---:|---:|
| A | 0.217 s | 0.77 ms | 0.986 | 0 % |
| B | 0.163 s | 0.96 ms | 0.998 | 35.1 % |
| C | 0.148 s | 0.97 ms | 0.997 | 0 % |

B and C cost the same per token. B pads its three sequences to the longest one and processes 35 % more
tokens; that, plus the fixed cost per call that C amortizes over three questions, is the entire 1.39x. Any
engine that batches without padding (vLLM, SGLang, Transformers with variable-length attention) would erase
it for unrelated questions.

The reason packing could save compute at all is structural, not numerical: if N questions share a context
of P tokens and each question is q tokens, batching processes N x (P + q) tokens and packing processes
P + N x q. With no shared context, P = 0 and there is nothing to save. MMLU questions share nothing, so the
pilot could never have shown a real gain. It had to be redesigned around a shared state.

## 4. The corrected experiment

Two runs, larger, with the padding confound controlled (`results/v2_race_32679039/`,
`results/v2_mmlu_32679038/`).

**RACE-H, 250 passages with exactly 4 questions each (n = 1000).** One passage, four multiple-choice
questions: the shared-state case. In C the passage appears only in the first turn; later turns carry only
the question. B's padding is small here (2.8 %) because the passages dominate sequence length.

| Mode | Accuracy | Questions / s | Tokens processed |
|---|---:|---:|---:|
| A: one at a time | 92.6 % | 1.66 | 468,583 |
| B: batch of 4 | 92.8 % | 2.00 | 481,924 |
| C: 4 packed, passage once | 92.9 % | 4.55 | 186,898 |
| C, rotated order | 93.6 % | 4.57 | 186,898 |

C processes 2.5x fewer tokens than A or B and answers 2.3x more questions per second than batching. The
cost per token is again the same in every mode (0.95 to 0.97 ms), so this gain is structural and survives
the choice of engine.

Accuracy: C minus A = +0.3 points, 95 % bootstrap interval over passages from -0.9 to +1.4. C minus B =
+0.1, interval -1.0 to +1.2. The apparent ordering 92.6, 92.8, 92.9 is three questions out of a thousand;
B, which uses the identical prompt as A in a padded batch, already moves 15 answers on its own.

**MMLU, 1200 questions (the pilot's 300 first, plus 900 new), groups of 3, 6 and 12.** The no-shared-state
case, used to see how far packing can go. On the pilot's 300 questions, A reproduces 84.33 % exactly.

| Mode | Accuracy | Difference vs A (95 % CI) | Questions / s | Tokens processed |
|---|---:|---:|---:|---:|
| A: one at a time | 84.2 % | | 3.10 | 163,032 |
| A_pad: A right-padded to 1024 tokens | 83.9 % | -0.25 (-1.1 to +0.7) | 0.91 | 1,228,800 |
| B: batch of 3 | 83.8 % | -0.42 (-1.2 to +0.3) | 3.88 | 253,638 |
| C: 3 packed | 84.0 % | -0.17 (-1.3 to +1.0) | 5.44 | 165,432 |
| C: 3 packed, rotated | 83.3 % | -0.92 (-2.1 to +0.3) | 5.46 | 165,432 |
| C: 6 packed | 84.9 % | +0.75 (-0.6 to +2.2) | 6.21 | 166,032 |
| C: 12 packed | 84.2 % | 0.00 (-1.8 to +1.7) | 6.71 | 166,332 |

No mode differs from A beyond noise, up to twelve questions per sequence. With 1200 questions the interval
is about +/-1.2 points, so drops smaller than that remain possible. C is faster than A and B here for the
reasons already identified: padding waste in B (35.7 %) and amortization of the 0.15 to 0.22 s fixed cost
per call. It is not cheaper per token.

## 5. Interference: accuracy holds, individual answers move

Later questions in a packed sequence can attend to earlier questions and to the placeholders between
them. To measure the effect honestly we needed a noise floor, because int8 kernels give slightly different
logits for the same prefix when the batch has a different shape.

`A_pad` provides that floor: the exact prompt of A, one question at a time, right-padded to a fixed
length. Nothing else in the context, only the tensor shape changes. It flips 32 of 1200 answers (2.7 %)
with a mean absolute probability change of 0.0125. B, same prompt in a padded batch, flips 2.3 % with 0.0126.
That is the noise.

Packing is three times above it:

| Mode and position | Answers changed vs A | Mean abs. probability change |
|---|---:|---:|
| A_pad (noise floor) | 2.7 % | 0.013 |
| B, any position | 2.3 % | 0.013 |
| C: 3 packed, position 0 (same prefix as A) | 2.5 % | 0.013 |
| C: 3 packed, positions 1 and 2 | 8.0 % | 0.036 |
| C: 6 packed, all | 7.4 % | 0.043 |
| C: 12 packed, all | 8.7 % | 0.050 |

Position 0 of a packed sequence has exactly A's causal prefix and sits on the noise floor, which is also a
check that the readout indices are right. On RACE-H the flips grow with distance from the passage: 1, 9, 10
and 19 out of 250 at positions 0 to 3. Rotating the question order changes 8.2 % of MMLU answers and 2.4 %
of RACE-H answers.

The changes are symmetric: as many answers go from wrong to right as from right to wrong, which is why
aggregate accuracy does not move. But a specific decision can depend on which questions accompany it and
in what order. For a library that means: `packed` when you care about throughput on a shared state,
`separate` when a decision must not change with the batch it arrived in.

## 6. The finding: small models pay for packing

The library was then run end to end, with both backends, on a smaller model: `Qwen/Qwen3.5-4B` in BF16,
RACE-H, 100 passages x 4 questions, one H200 MIG 3g.71gb slice, wall-clock time including tokenization
(`results/lib_race_32698399/`). `separate` is one sequence per question with the full passage; on vLLM it
uses constrained one-token generation with `allowed_token_ids` and the engine's prefix cache.

| Backend, mode | Accuracy | Questions / s | Tokens sent | Agreement with Transformers `separate` |
|---|---:|---:|---:|---:|
| Transformers, `separate` | 87.3 % | 20.0 | 179,807 | |
| Transformers, `packed` | 84.5 % | 49.6 | 71,216 | 93.8 % |
| vLLM, `separate` | 87.0 % | 71.9 | 179,807 | 99.8 % |
| vLLM, `packed` | 84.3 % | 38.7 | 71,216 | 93.8 % |

Two things the 27B runs did not show.

**Packing costs the 4B model 2.8 points** (11 of 400 answers) where the 27B lost nothing. Interference is
the same mechanism, but a smaller model is less able to keep four questions apart. This is the most useful
caveat in the project: the "accuracy holds" result is a property of the model size, not of the method.

**On vLLM, `separate` is the fastest mode**, 1.9x faster than packed, with no accuracy cost. The prefix
cache already encodes the shared passage once, which is the saving packing was designed to capture, and
reading exact label scores from one constrained token is cheaper than extracting top-k `prompt_logprobs` at
every interior position. Packing's throughput advantage is real on Transformers (2.5x here, no prefix
cache) and gone on an engine that has one.

The two backends agree: same answer 99.8 % of the time in `separate` mode, mean absolute probability
difference 0.003.

## 7. Calibration: one scalar

Raw confidence is too high. On MMLU with the 27B, mean confidence is 0.896 against 0.842 accuracy. We
fitted a single temperature by cross-validation: fit on one half, evaluate on the other, and the reverse,
never scoring the fold used for fitting.

| Run and mode | Fitted T | ECE raw | ECE scaled |
|---|---:|---:|---:|
| MMLU, A | 1.45 to 1.50 | 5.4 % | 2.1 % |
| MMLU, C: 12 packed | 1.30 | 3.8 % | 1.2 % |
| RACE-H, A | 1.30 to 1.35 | 2.8 % | 1.1 % |
| RACE-H, C: 4 packed | 1.00 to 1.10 | 1.1 % | 1.6 % |

Temperature does not change which option wins; it only reshapes the probabilities. One fitted scalar
brings the expected calibration error from 5.4 % to 2.1 % on MMLU. Packed modes come out less
over-confident than one-at-a-time scoring in every run (raw ECE 3.8 % vs 5.4 % on MMLU, 1.1 % vs 2.8 % on
RACE-H); we have no established explanation, and with 1000 to 1200 examples the ECE is noisy enough that the
direction is clear and the magnitude is not. RACE-H packed was already near calibrated, and scaling does
not improve it.

## What this does and does not establish

- Reading several typed answers from one forward pass keeps aggregate accuracy on a 27B model up to 12
  questions with no shared state and 4 with a shared passage, within about a point. On a 4B model it costs
  about 3 points.
- The compute saving is structural and only exists with a shared state; it is captured equally well by a
  prefix cache. Without one, packing is 2.5x on Transformers. With one, keep questions separate.
- Individual answers depend on their neighbours in 6 to 9 % of cases, three times the numerical noise
  floor, without moving accuracy.
- Raw probabilities are over-confident by about 5 points and one cross-validated scalar fixes most of it.
- Not measured: a generation-with-reasoning baseline (the comparison TypeSafe's charts make), other model
  families, BF16 versus int8 for the same checkpoint, and anything about how Jev itself works. The pilot
  and the analysis that corrected it are kept in `results/mmlu_32677666/` on purpose.

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

**Packing costs the 4B model 2.8 points** (11 of 400 answers) on this 100-passage subset, where the 27B lost
nothing. On all 250 passages (section 9) the 4B loses nothing either, while 0.6B to 2B models lose 2 to 8
points, so the 4B sits on the boundary. Interference is the same mechanism at every size; a smaller model is
less able to keep four questions apart. The "accuracy holds" result is a property of the model size, not of
the method.

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

## 8. The comparison people asked for: Jev and Laya on one benchmark

After the write-up above went public, the obvious question was how this compares with Jev itself and with
Laya, the 421M ModernBERT-based reproduction that claims to beat Jev. Laya's own headline (83.8 % vs 67.8 %)
compares two different benchmarks, which a third party (Luni/laya-jev-benchmark) had already pointed out. The
shared benchmark where published numbers exist is `LocalLLaMA/typed-decisions`: 400 test cases, each a JSON
state plus five typed questions (`noul` yes/no, `choice`, ordered `score`), 2,000 decisions, gold = the mean of
three samples from a roughly 4B-class teacher. Accuracy therefore measures agreement with that teacher; the
card puts the majority baseline at 0.520, "perfect scenario understanding" at 0.704 and teacher
self-agreement at 0.735, and warns that scores much above 0.75 mean a model learned the teacher's quirks.

We wrote an adapter (`benchmarks/scripts/typed_decisions.py`) that renders each case as a state plus five
so1 `Choice` questions (option descriptions included in the prompt, as the card requires), and ported the
third-party scorer line by line. Then we ran Laya through it first. It reproduced Laya's published numbers
exactly: base checkpoint 0.360 accuracy / 0.176 ECE (published 0.360 / 0.175), fine-tuned checkpoint 0.766
on CPU and 0.769 on the GPU / 0.216 ECE (published 0.766 / 0.212). With the scorer validated, the stock Qwen
models went through the same code, zero-shot (`results/td_*`).

| Model | Accuracy | ECE | Brier | KL to gold | ms / case |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B, packed | 29.1 % | 0.534 | 0.672 | 2.25 | 46 |
| Qwen3-0.6B-FP8, packed (vLLM) | 30.7 % | 0.530 | 0.681 | 2.28 | 77 |
| Qwen3-0.6B BF16, packed (vLLM) | 29.1 % | 0.536 | 0.677 | 2.26 | 71 |
| Qwen3-1.7B, packed | 45.9 % | 0.509 | 0.682 | 4.25 | 55 |
| Qwen3.5-2B, packed | 47.3 % | 0.124 | 0.269 | 0.50 | 85 |
| Qwen3.5-4B, packed | 59.3 % | 0.118 | 0.164 | 0.38 | 105 |
| Qwen3.5-4B, separate | 56.0 % | 0.215 | 0.262 | 0.55 | 229 |
| Laya base (measured) | 36.0 % | 0.176 | 0.329 | 0.55 | 23 |
| Jev 1.13.0 (measured by the benchmark authors via TypeSafe's API, 2026-09-18) | 72.7 % | 0.144 | 0.148 | 1.44 | 710 |
| Qwen3.6-27B 8-bit, separate | 72.7 % | 0.063 | 0.120 | 0.36 | 1234 |
| **Qwen3.6-27B 8-bit, packed** | **73.7 %** | **0.020** | 0.113 | 0.27 | 582 |
| Teacher self-agreement | 73.5 % | | | | |
| Laya fine-tuned on this benchmark (measured) | 76.9 % | 0.216 | 0.066 | 0.12 | 23 |

What it says:

- A stock 27B with nothing trained lands on the teacher ceiling and one point above the Jev row. That row is
  not a TypeSafe publication: the benchmark's authors ran all 400 cases through TypeSafe's API on 2026-09-18
  (`jev-latest`, reporting itself as 1.13.0; p50 710 ms per case, $0.016 in total). On the distribution
  metrics the card asks readers to prefer over ECE, the gap is large: KL to gold 0.27 against Jev's 1.44,
  Brier 0.113 against 0.148. Jev commits hard to the right label; the stock 27B reproduces the teacher's
  spread.
  Per workflow it is 65.0 % on agent-trace observability (the hardest, teacher ceiling 0.56 on its urgency
  question), 77.6 % customer service, 77.4 % invoice processing, 74.6 % security incidents.
- Its ECE of 0.020 is the lowest in the table by a wide margin, with the caveat from the card that a base-rate
  prior which reads nothing scores ECE 0.088. Fine-tuned Laya gets the highest accuracy,
  above the ceiling, by fitting the teacher's distribution (best Brier and KL) while its argmax confidence is
  badly calibrated. Those are two different things to be good at, and the benchmark card asks for both.
- Packing helps on this benchmark for every model of 2B and up (27B +1.0 point, 4B +3.3 points, 2B is the
  exception at -2.9), the opposite of the RACE-H result on the 4B. Five questions about the same JSON state
  are related, and seeing the others appears to help; interference is not always a cost.
- Below 4B, zero-shot decisions collapse and confidence stays high (Qwen3-1.7B: 45.9 % at ECE 0.51). FP8
  weights do not move the needle: Qwen3-0.6B in FP8 and BF16 through the same vLLM engine differ by a point.
  (Transformers could not load the FP8 checkpoint on the cluster, whose compute nodes have no nvcc for its JIT
  kernels; vLLM ran it with DeepGEMM disabled.) Laya
  base, which has never seen the task, scores 36 %. Small models need training for this; large ones do not.
- Latencies are wall-clock per case on one H200 MIG slice. Laya is a 421M encoder and is 25x faster than
  the 27B; the 27B here runs 8-bit with fallback kernels for its linear-attention layers, so its absolute
  time is far from what a served deployment would see. Jev's 710 ms is the API round-trip the benchmark
  authors measured, network included. The two are the same order of magnitude and not a controlled race.
- Temperature fitted on 200 train cases against the soft gold (T = 2.0 for the 27B) improves Brier 0.113 to
  0.074 and KL 0.27 to 0.15 but raises hard-label ECE from 0.020 to 0.135: matching a soft teacher
  distribution and being calibrated on the argmax are different objectives. Both sets of numbers are in
  `results/`.

## 9. RACE-H by model size, and Laya on reading comprehension

The same 250 passages x 4 questions and the same four modes, for every stock Qwen we could fit on a 35 GB
slice, plus Laya's two checkpoints answering the four questions of a passage in one call (`results/race_*`).

| Model | A | B4 | C4 (packed) | C4 - A | Answers changed by packing | Order change | ECE (C4) | Questions / s (C4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 50.0 % | 49.8 % | 42.5 % | -7.5 | 36.9 % | 37.9 % | 0.32 | 134 |
| Qwen3-1.7B | 69.9 % | 70.4 % | 68.0 % | -1.9 | 20.0 % | 18.8 % | 0.29 | 100 |
| Qwen3.5-2B | 77.0 % | 77.0 % | 75.2 % | -1.8 | 13.9 % | 12.9 % | 0.03 | 53 |
| Qwen3.5-4B | 84.8 % | 84.8 % | 85.0 % | +0.2 | 7.6 % | 7.6 % | 0.04 | 27 |
| Qwen3.6-27B, 8-bit | 92.6 % | 92.8 % | 92.9 % | +0.3 | 3.9 % | 2.4 % | 0.01 | 4.6 |
| Laya base (421M, own packed call) | | | 44.6 % | | | | 0.05 | 56 |
| Laya fine-tuned on typed-decisions | | | 45.9 % | | | | 0.08 | 57 |

This is the cleanest statement of the interference result. The fraction of answers that move when a question
is packed with its neighbours falls monotonically with size, from 37 % at 0.6B to 3.9 % at 27B, and so does
the accuracy cost: 7.5 points at 0.6B, about 2 points at 1.7B and 2B, none at 4B and 27B. The 4B is the
boundary: on a 100-passage subset (section 6) it had lost 2.8 points, on all 250 passages it loses none, and its
7.6 % of moved answers is still twice the 27B's. Batching (B4) never costs anything, as before.

The Qwen3 generation (0.6B, 1.7B) is also badly over-confident on this task (ECE 0.27 to 0.33), where the
Qwen3.5 models are within 0.05 raw. Whatever changed in post-training between the two generations matters
more for calibration than model size does.

Laya, run on RACE-H through its own choice interface, scores 44.6 % (base) and 45.9 % (the checkpoint tuned on
typed-decisions) on four-option questions, below a stock 0.6B decoder read one question at a time. Only 10 of
250 passages exceed its 512-token context, so truncation is not the reason; it is a decision encoder trained on
short states, and reading comprehension over a 400-token passage is a different task. It is, however, fast (56
questions per second, wall-clock) and reasonably calibrated here (ECE 0.046). None of this is a criticism of
Laya on its own tasks; it is the reason a "which model" question needs a "for what" attached.

## 10. Option order: the bias we had not measured, and the fix

An external leaderboard that ran this library reported that reversing the two options of yes/no questions
("A. no, B. yes" instead of "A. yes, B. no") dropped its accuracy from 72 % to 21 %. We had measured
sensitivity to the order of *questions* (section 5) but never to the order of *options*, which is a
different and older problem: a model that answers with a letter can prefer the letter.

So we measured it, and shipped the standard fix as an option. `permutations=2` presents every question
twice, once as given and once reversed, and averages the two probability vectors, mapped back to the
caller's order; `permutations=k` adds cyclic shifts. Three configurations, two models, two benchmarks,
packed mode (`results/oo_*`):

| Benchmark, model | As given | Reversed | Averaged (`permutations=2`) | Latency, averaged |
|---|---:|---:|---:|---:|
| typed-decisions, Qwen3.5-4B, accuracy | 59.3 % | 55.2 % | 59.5 % | 177 ms/case (1.7x) |
| typed-decisions, Qwen3.5-4B, yes/no only | 76.7 % | 63.2 % | 70.7 % | |
| typed-decisions, Qwen3.5-4B, ECE / KL / Brier | 0.118 / 0.38 / 0.164 | 0.131 / 0.44 / 0.225 | 0.062 / 0.29 / 0.152 | |
| typed-decisions, Qwen3.6-27B, accuracy | 73.7 % | 75.3 % | 75.5 % | 969 ms/case (1.7x) |
| typed-decisions, Qwen3.6-27B, yes/no only | 82.7 % | 83.0 % | 84.3 % | |
| typed-decisions, Qwen3.6-27B, ECE / KL / Brier | 0.020 / 0.27 / 0.113 | 0.029 / 0.29 / 0.120 | 0.0075 / 0.23 / 0.098 | |
| RACE-H (100 passages), Qwen3.5-4B, accuracy / ECE | 84.5 % / 0.037 | 84.5 % / 0.042 | 85.5 % / 0.029 | 39 ms/question (1.9x) |
| RACE-H (100 passages), Qwen3.6-27B, accuracy / ECE | 94.0 % / 0.021 | 92.8 % / 0.014 | 93.5 % / 0.012 | 215 ms/question (1.9x) |

What it says:

- The bias exists and scales inversely with model size, like interference did. Reversing the options costs
  the 4B four points overall and 13.5 points on yes/no questions, where the first position ("yes") is
  favoured. The 27B moves 1.7 points, and in the other direction: on these prompts it slightly prefers the
  second position for choice questions (67.8 % as given, 73.8 % reversed). Neither is the 51-point collapse
  the leaderboard reported; whatever prompt they ran produced a much larger effect than ours does.
- Averaging is the right default for anyone who consumes the probabilities. On typed-decisions it improves
  every metric for both models, and for the 27B it gives the best distribution numbers of any zero-shot
  configuration we have: 75.5 % accuracy, ECE 0.0075, KL 0.23, Brier 0.098, all with no training. On RACE-H,
  four options and a long passage, the two orders already agree and averaging changes nothing beyond noise.
- The price is two forward passes per item: 1.7x latency in packed mode (the second pass shares nothing
  with the first in Transformers; on vLLM the prefix cache would absorb the shared state).

We left it off by default. The library's baseline numbers stay reproducible as published, and the option
is one argument away for anyone who wants position-invariant probabilities and can pay for them.

## What this does and does not establish

- Reading several typed answers from one forward pass keeps aggregate accuracy on a 27B model up to 12
  questions with no shared state and 4 with a shared passage, within about a point. On a 4B model it costs
  about 3 points.
- The compute saving is structural and only exists with a shared state; it is captured equally well by a
  prefix cache. Without one, packing is 2.5x on Transformers. With one, keep questions separate.
- Individual answers depend on their neighbours in 6 to 9 % of cases, three times the numerical noise
  floor, without moving accuracy.
- Raw probabilities are over-confident by about 5 points and one cross-validated scalar fixes most of it.
- Option order matters, more for small models; asking in both orders and averaging (`permutations=2`)
  removes it at twice the compute and, on typed-decisions, improves every metric.
- On the community benchmark for this task, a stock 27B zero-shot matches Jev's measured accuracy with
  probabilities far closer to the gold distribution; a small model fine-tuned on the benchmark scores higher but past the point where
  the benchmark's authors say the score means anything.
- Not measured: a generation-with-reasoning baseline (the comparison TypeSafe's charts make), other model
  families, BF16 versus int8 for the same checkpoint, and anything about how Jev itself works. The pilot
  and the analysis that corrected it are kept in `results/mmlu_32677666/` on purpose.

# MMLU pilot results

Run: `mmlu_32677666`. Complete.

| Mode | Questions | Accuracy | Questions / GPU forward second | Peak allocated GiB |
|---|---:|---:|---:|---:|
| A | 300 | 84.33% | 3.139 | 28.17 |
| B | 300 | 84.00% | 4.009 | 28.56 |
| C | 300 | 84.00% | 5.573 | 28.22 |
| C_rotated | 300 | 84.33% | 5.572 | 28.23 |

First-position C vs A maximum probability difference: 0.154443. These share identical causal prefixes; nonzero differences can reflect kernels/quantization.

This measures forced A/B/C/D decisions, with no generated reasoning. Probabilities are normalized over these four labels and are not calibrated. Timing excludes load, tokenization and tensor preparation. C uses concatenated chat turns and fixed placeholders. All modes use the same quantized checkpoint. These are pilot results, not a reproduction of published MMLU scores.

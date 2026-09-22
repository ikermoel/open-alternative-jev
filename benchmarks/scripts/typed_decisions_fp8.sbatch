#!/bin/bash
#SBATCH --job-name=so1-td-fp8
#SBATCH --account=ucb-general
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --exclude=c3gpu-e5-u5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/ikmo5391/so1/logs/tdfp8_%j.log
# Qwen3-0.6B-FP8 through vLLM (native FP8, no JIT kernels), and Qwen3-0.6B BF16 through vLLM as the matched control.
set -uo pipefail
trap 'rc=$?; echo TD_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/so1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache VLLM_CACHE_ROOT=$P/cache/vllm
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=$R VLLM_USE_FLASHINFER_SAMPLER=0
# Compute nodes have no nvcc: DeepGEMM JIT-compiles FP8 GEMMs, so disable it and let vLLM use its prebuilt FP8 kernels.
export VLLM_USE_DEEP_GEMM=0
cd "$R"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
DATA=$(dirname "$(ls $P/hf/hub/datasets--LocalLLaMA--typed-decisions/snapshots/*/all/test-00000-of-00001.parquet | head -1)")
PY=$P/venv-vllm/bin/python
J=$SLURM_JOB_ID
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -E "CASES|TEMPERATURE|PROGRESS 400|RESULT|Traceback|Error" | cut -c1-400; }

for name in ${MODELS:-Qwen3-0.6B-FP8 Qwen3-0.6B}; do
  M=$(ls -d $P/hf/hub/models--Qwen--$name/snapshots/* | head -1)
  tag=$(echo "$name" | tr 'A-Z.' 'a-z_')
  run "$name vllm packed"   $PY benchmarks/scripts/typed_decisions.py --backend so1 --engine vllm --gpu-util 0.6 --model "$M" --mode packed   --data-dir "$DATA" --out results/td_${tag}_vllm_packed_$J
  run "$name vllm separate" $PY benchmarks/scripts/typed_decisions.py --backend so1 --engine vllm --gpu-util 0.6 --model "$M" --mode separate --data-dir "$DATA" --out results/td_${tag}_vllm_separate_$J
done

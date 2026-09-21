#!/bin/bash
#SBATCH --job-name=so1-td-small
#SBATCH --account=ucb-general
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --exclude=c3gpu-e5-u5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/ikmo5391/so1/logs/tdsmall_%j.log
# typed-decisions with small models: Qwen3-0.6B-FP8, Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-2B. Same adapter and scorer.
set -uo pipefail
trap 'rc=$?; echo TD_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/so1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=$R
cd "$R"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
DATA=$(dirname "$(ls $P/hf/hub/datasets--LocalLLaMA--typed-decisions/snapshots/*/all/test-00000-of-00001.parquet | head -1)")
PY=$P/venv-vllm/bin/python
J=$SLURM_JOB_ID
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -E "CASES|TEMPERATURE|PROGRESS 400|RESULT|Traceback|Error" | cut -c1-400; }

for name in Qwen3-0.6B-FP8 Qwen3-0.6B Qwen3-1.7B Qwen3.5-2B; do
  M=$(ls -d $P/hf/hub/models--Qwen--$name/snapshots/* 2>/dev/null | head -1)
  [ -z "$M" ] && { echo "=== $name: not in cache, skipped"; continue; }
  tag=$(echo "$name" | tr 'A-Z.' 'a-z_')
  run "$name packed"         $PY benchmarks/scripts/typed_decisions.py --backend so1 --model "$M" --mode packed   --data-dir "$DATA" --out results/td_${tag}_packed_$J
  run "$name separate"       $PY benchmarks/scripts/typed_decisions.py --backend so1 --model "$M" --mode separate --data-dir "$DATA" --out results/td_${tag}_separate_$J
  run "$name packed cal200"  $PY benchmarks/scripts/typed_decisions.py --backend so1 --model "$M" --mode packed --calibrate 200 --data-dir "$DATA" --out results/td_${tag}_packed_cal_$J
done

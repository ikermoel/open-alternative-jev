#!/bin/bash
#SBATCH --job-name=so1-order
#SBATCH --account=ucb-general
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --exclude=c3gpu-e5-u5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/ikmo5391/so1/logs/order_%j.log
# Option-order sensitivity and the permutations=2 fix: typed-decisions and RACE-H, Qwen3.5-4B and Qwen3.6-27B.
# Three configurations per model and benchmark: options as given, options reversed, both orders averaged.
set -uo pipefail
trap 'rc=$?; echo OO_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/so1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=$R
cd "$R"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
DATA=$(dirname "$(ls $P/hf/hub/datasets--LocalLLaMA--typed-decisions/snapshots/*/all/test-00000-of-00001.parquet | head -1)")
Q4B=$(ls -d $P/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1)
Q27B=$(cat $P/open-jev/data/model_path.txt)
NEW=$P/venv-vllm/bin/python
OLD=$P/open-jev/venv/bin/python
J=$SLURM_JOB_ID
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -E "CASES|ITEMS|RESULT|Traceback|Error" | cut -c1-600; }

for cfg in "original 1" "reversed 1" "original 2"; do
  set -- $cfg; order=$1; perm=$2; tag=${order}_p${perm}
  run "typed-decisions 4B $tag"  $NEW benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q4B" --mode packed --option-order $order --permutations $perm --data-dir "$DATA" --out results/oo_td_qwen4b_${tag}_$J
  run "typed-decisions 27B $tag" $OLD benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q27B" --load-8bit --mode packed --option-order $order --permutations $perm --data-dir "$DATA" --out results/oo_td_qwen27b_${tag}_$J
  run "RACE-H 4B $tag"  $NEW benchmarks/scripts/library_compare.py --model "$Q4B" --backends hf --modes packed --passages 100 --option-order $order --permutations $perm --out benchmarks/results/oo_race_qwen4b_${tag}_$J
  run "RACE-H 27B $tag" $OLD benchmarks/scripts/library_compare.py --model "$Q27B" --hf-8bit --backends hf --modes packed --passages 100 --option-order $order --permutations $perm --out benchmarks/results/oo_race_qwen27b_${tag}_$J
done

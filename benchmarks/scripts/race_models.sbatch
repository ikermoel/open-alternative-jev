#!/bin/bash
#SBATCH --job-name=so1-race-models
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
#SBATCH --output=/projects/ikmo5391/so1/logs/racemodels_%j.log
# RACE-H (250 passages x 4) with the small Qwen models through benchmark_v2 (modes A, B4, C4, C4_rot) and with Laya.
set -uo pipefail
trap 'rc=$?; echo RM_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/so1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=$R
cd "$R/benchmarks"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
PY=$P/venv-vllm/bin/python
LAYA=$(ls -d $P/hf/hub/models--convaiinnovations--laya/snapshots/* | head -1)
J=$SLURM_JOB_ID
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -E "PASSAGES|WARMUP_OK|RESULT|SUMMARY|Traceback|Error" | cut -c1-500; }

run "laya base"            $PY scripts/race_laya.py --model "$LAYA" --out results/race_laya_base_$J
run "laya typed-decisions" $PY scripts/race_laya.py --model "$LAYA/typed-decisions" --out results/race_laya_ft_$J
for name in Qwen3-0.6B Qwen3-1.7B Qwen3.5-2B Qwen3.5-4B; do
  M=$(ls -d $P/hf/hub/models--Qwen--$name/snapshots/* | head -1)
  tag=$(echo "$name" | tr 'A-Z.' 'a-z_')
  run "$name RACE-H A/B4/C4/C4_rot" $PY scripts/benchmark_v2.py --data race1000.jsonl --group-size 4 --modes A,B4,C4,C4_rot \
      --model "$M" --no-quant --device cuda --out results/race_${tag}_$J
  $PY scripts/analyze_run.py results/race_${tag}_$J > results/race_${tag}_$J/analysis.txt 2>&1
done

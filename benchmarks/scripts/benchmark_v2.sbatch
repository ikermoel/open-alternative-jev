#!/bin/bash
#SBATCH --job-name=openjev-v2
#SBATCH --account=ucb-general
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/ikmo5391/open-jev/logs/v2_%j.log
# Submit with:  sbatch --export=ALL,DATASET=mmlu scripts/benchmark_v2.sbatch
#          or:  sbatch --export=ALL,DATASET=race scripts/benchmark_v2.sbatch
set -euo pipefail
trap 'rc=$?; echo BENCHMARK_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/open-jev
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
cd "$R"
nvidia-smi --query-gpu=name,memory.total --format=csv
case "${DATASET:?set DATASET=mmlu or race}" in
  mmlu) ARGS="--data mmlu1200.jsonl --group-size 12 --modes A,A_pad,B3,C3,C3_rot,C6,C12" ;;
  race) ARGS="--data race1000.jsonl --group-size 4 --modes A,B4,C4,C4_rot" ;;
  *) echo "unknown DATASET=$DATASET"; exit 2 ;;
esac
echo "DATASET=$DATASET ARGS=$ARGS"
"$R/venv/bin/python" -u scripts/benchmark_v2.py $ARGS --out "results/v2_${DATASET}_${SLURM_JOB_ID}"
"$R/venv/bin/python" -u scripts/analyze_run.py "results/v2_${DATASET}_${SLURM_JOB_ID}"

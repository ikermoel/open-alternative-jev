#!/bin/bash
#SBATCH --job-name=openjev-mmlu
#SBATCH --account=ucb-general
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/ikmo5391/open-jev/logs/mmlu_%j.log
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
"$R/venv/bin/python" -u scripts/benchmark.py --count 300 --out "results/mmlu_${SLURM_JOB_ID}"

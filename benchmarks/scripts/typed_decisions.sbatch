#!/bin/bash
#SBATCH --job-name=so1-td
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
#SBATCH --output=/projects/ikmo5391/so1/logs/td_%j.log
# typed-decisions: Laya (base, fine-tuned) and so1 (Qwen3.5-4B, Qwen3.6-27B) on the same GPU, same scorer.
set -uo pipefail
trap 'rc=$?; echo TD_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/so1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=$R
cd "$R"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
DATA=$(dirname "$(ls $P/hf/hub/datasets--LocalLLaMA--typed-decisions/snapshots/*/all/test-00000-of-00001.parquet | head -1)")
LAYA=$(ls -d $P/hf/hub/models--convaiinnovations--laya/snapshots/* | head -1)
Q4B=$(ls -d $P/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1)
Q27B=$(cat $P/open-jev/data/model_path.txt)
J=$SLURM_JOB_ID
NEW=$P/venv-vllm/bin/python          # torch 2.13, transformers 5.17, laya
OLD=$P/open-jev/venv/bin/python      # torch 2.6, transformers 5.16, bitsandbytes (for the 27B in 8-bit)
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -E "CASES|TEMPERATURE|PROGRESS|RESULT|Traceback|Error" | cut -c1-400; }

run "laya base"            $NEW benchmarks/scripts/typed_decisions.py --backend laya --model "$LAYA" --data-dir "$DATA" --out results/td_laya_base_$J
run "laya typed-decisions" $NEW benchmarks/scripts/typed_decisions.py --backend laya --model "$LAYA/typed-decisions" --data-dir "$DATA" --out results/td_laya_ft_$J
run "so1 4B packed"        $NEW benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q4B" --mode packed   --data-dir "$DATA" --out results/td_qwen4b_packed_$J
run "so1 4B separate"      $NEW benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q4B" --mode separate --data-dir "$DATA" --out results/td_qwen4b_separate_$J
run "so1 4B packed cal200" $NEW benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q4B" --mode packed --calibrate 200 --data-dir "$DATA" --out results/td_qwen4b_packed_cal_$J
run "so1 27B packed"       $OLD benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q27B" --load-8bit --mode packed   --data-dir "$DATA" --out results/td_qwen27b_packed_$J
run "so1 27B separate"     $OLD benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q27B" --load-8bit --mode separate --data-dir "$DATA" --out results/td_qwen27b_separate_$J
run "so1 27B packed cal200" $OLD benchmarks/scripts/typed_decisions.py --backend so1 --model "$Q27B" --load-8bit --mode packed --calibrate 200 --data-dir "$DATA" --out results/td_qwen27b_packed_cal_$J

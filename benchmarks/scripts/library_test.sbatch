#!/bin/bash
#SBATCH --job-name=so1-lib
#SBATCH --account=ucb-general
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/ikmo5391/so1/logs/lib_%j.log
# Library test on the cluster: vLLM engine smoke, pytest (HF + vLLM backends), quickstart, RACE-H comparison.
set -uo pipefail
trap 'rc=$?; echo LIBTEST_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/so1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache VLLM_CACHE_ROOT=$P/cache/vllm
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=$R
# Compute nodes have no nvcc/ninja: avoid every JIT path (FlashInfer sampler, FlashInfer GDN prefill kernel).
export VLLM_USE_FLASHINFER_SAMPLER=0
export SO1_VLLM_KWARGS='{"gdn_prefill_backend": "triton"}'
cd "$R"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
MODEL=$(ls -d $P/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1)
export SO1_TEST_MODEL=$MODEL SO1_GPU_UTIL=0.45
PY=$P/venv-vllm/bin/python
$PY -c "import vllm, torch, transformers; print('VERSIONS', vllm.__version__, torch.__version__, transformers.__version__, 'cuda', torch.cuda.is_available())"

echo "=== VLLM SMOKE ==="
$PY benchmarks/scripts/vllm_smoke.py > logs/vllm_smoke_${SLURM_JOB_ID}.log 2>&1
grep -E "VLLM_SMOKE_OK" logs/vllm_smoke_${SLURM_JOB_ID}.log || { echo "VLLM_SMOKE_FAILED; last 60 lines:"; grep -vE "^\s*$" logs/vllm_smoke_${SLURM_JOB_ID}.log | tail -60 | cut -c1-300; }

sleep 20  # let the vLLM worker processes release the MIG device before the next CUDA context (NVML "Unknown Error" seen otherwise)
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1
echo "=== PYTEST ==="
$PY -m pytest -q -p no:cacheprovider tests > logs/pytest_${SLURM_JOB_ID}.log 2>&1
echo "PYTEST_EXIT=$?"
tail -12 logs/pytest_${SLURM_JOB_ID}.log | cut -c1-200

echo "=== QUICKSTART hf ==="
SO1_MODEL=$MODEL SO1_BACKEND=hf $PY examples/quickstart.py 2>&1 | grep -vE "Warning|Loading weights" | tail -8

echo "=== LIBRARY COMPARE ==="
$PY -u benchmarks/scripts/library_compare.py --model "$MODEL" --passages ${PASSAGES:-100} --backends ${BACKENDS:-hf,vllm} --gpu-util 0.85 --vllm-kwargs "$SO1_VLLM_KWARGS" --out "benchmarks/results/lib_race_${SLURM_JOB_ID}" > logs/compare_${SLURM_JOB_ID}.log 2>&1
echo "COMPARE_EXIT=$?"
grep -E "ITEMS|RESULT|SUMMARY|Traceback|Error" logs/compare_${SLURM_JOB_ID}.log | cut -c1-600

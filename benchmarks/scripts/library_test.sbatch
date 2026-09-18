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
# Library test on the cluster: pytest (HF + vLLM backends) and the RACE-H comparison with Qwen3.5-4B.
set -uo pipefail
trap 'rc=$?; echo LIBTEST_EXIT=$rc' EXIT
P=/projects/ikmo5391
R=$P/so1
export HF_HOME=$P/hf HF_HUB_OFFLINE=1 XDG_CACHE_HOME=$P/cache VLLM_CACHE_ROOT=$P/cache/vllm
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH=$R
unset PYTHONPATH_OLD
cd "$R"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
MODEL=$(ls -d $P/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/* | head -1)
export SO1_TEST_MODEL=$MODEL SO1_GPU_UTIL=0.45
PY=$P/venv-vllm/bin/python
$PY -c "import vllm, torch, transformers; print('VERSIONS', vllm.__version__, torch.__version__, transformers.__version__)"
echo "=== PYTEST ==="
$PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -30
echo "PYTEST_EXIT=${PIPESTATUS[0]}"
echo "=== QUICKSTART hf ==="
SO1_MODEL=$MODEL SO1_BACKEND=hf $PY examples/quickstart.py 2>&1 | grep -vE "Warning|Loading weights" | tail -8
echo "=== LIBRARY COMPARE ==="
$PY -u benchmarks/scripts/library_compare.py --model "$MODEL" --passages ${PASSAGES:-100} --backends hf,vllm --gpu-util 0.85 --out "benchmarks/results/lib_race_${SLURM_JOB_ID}" 2>&1 | grep -E "ITEMS|RESULT|SUMMARY|Traceback|Error|error" | cut -c1-600

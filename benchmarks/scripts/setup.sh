#!/bin/bash
set -euo pipefail
P=/projects/ikmo5391
R=$P/open-jev
export HF_HOME=$P/hf XDG_CACHE_HOME=$P/cache
unset HF_HUB_OFFLINE
$P/bin/uv venv --python $P/venv/bin/python "$R/venv"
printf '%s\n' "$P/venv/lib/python3.11/site-packages" > "$R/venv/lib/python3.11/site-packages/cluster_base.pth"
$P/bin/uv pip install --python "$R/venv/bin/python" --no-deps bitsandbytes pyarrow
"$R/venv/bin/python" -u "$R/scripts/prepare.py"
echo SETUP_EXIT_0

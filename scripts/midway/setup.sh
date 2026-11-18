#!/bin/bash
# One-time setup on a Midway3 LOGIN node (needs internet; compute nodes may not have it).
#   bash scripts/midway/setup.sh
# Installs uv, builds .venv with the project + vLLM, and downloads the model weights to scratch.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SCRATCH_DIR="/scratch/midway3/$USER"
export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"   # keep large caches out of the 30 GB home quota
export HF_HOME="$SCRATCH_DIR/hf"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

cd "$(dirname "$0")/../.."                      # repo root

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --python 3.12                           # project deps (uv downloads Python 3.12 if needed)
uv pip install vllm                             # Linux/GPU only, so not in pyproject.toml

.venv/bin/python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download("$MODEL"))
EOF

echo "Setup done. Copy the final dataset to data/processed/, then: sbatch scripts/midway/score_panel.sbatch"

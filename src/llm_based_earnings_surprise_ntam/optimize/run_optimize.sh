#!/bin/bash
#SBATCH --job-name=ntam-gepa
#SBATCH --account=finm36000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gepa_%j.out
#
# GEPA prompt optimization on a Midway GPU. Starts Ollama on the node, then runs one command. Submit from the repo
# root (the folder that contains src/):
#
#   sbatch src/llm_based_earnings_surprise_ntam/optimize/run_optimize.sh check
#   sbatch src/llm_based_earnings_surprise_ntam/optimize/run_optimize.sh static     --run first
#   sbatch src/llm_based_earnings_surprise_ntam/optimize/run_optimize.sh continuous --run first
#
# A job that hits the time limit stops mid-run: submit the same line again and it resumes.
# The #SBATCH lines above are the only settings not in optimize.yaml (Slurm reads them before anything runs).
#
# One-time setup, from the repo root (an environment of its own, outside the repo):
#   uv venv ~/.venvs/ntam-optimize --python 3.12
#   uv pip install --python ~/.venvs/ntam-optimize/bin/python -r src/llm_based_earnings_surprise_ntam/optimize/requirements.txt
#   mkdir -p logs
# Override with CONFIG=... / PYTHON=... in front of sbatch.

set -euo pipefail
CMD=${1:?usage: run_optimize.sh check|static|continuous [--run NAME]}
shift
CONFIG=${CONFIG:-src/llm_based_earnings_surprise_ntam/optimize/optimize.yaml}
PYTHON=${PYTHON:-$HOME/.venvs/ntam-optimize/bin/python}
export PYTHONPATH=src PYTHONUNBUFFERED=1

# Ollama's binary, model folder and parallel slots, from optimize.yaml
eval "$("$PYTHON" -m llm_based_earnings_surprise_ntam.optimize.config --shell "$CONFIG")"
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))   # own port per job: nodes are shared

mkdir -p logs
OLOG="logs/ollama_${SLURM_JOB_ID}.log"
nvidia-smi
ollama serve > "$OLOG" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID' EXIT
for _ in $(seq 1 60); do
    ollama list > /dev/null 2>&1 && grep -q "inference compute" "$OLOG" && break
    sleep 2
done
ollama --version

# stop right away if Ollama runs on the CPU instead of the GPU (it would be ~50x slower)
if ! grep -i "inference compute" "$OLOG" | grep -qi "library=cuda"; then
    echo "ERROR: Ollama did not detect the GPU. See $OLOG"
    grep -iE "inference compute|driver|cuda" "$OLOG" | head
    exit 1
fi

"$PYTHON" -m "llm_based_earnings_surprise_ntam.optimize.$CMD" --config "$CONFIG" "$@"

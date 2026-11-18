#!/bin/bash
#SBATCH --job-name=ntam-grid-llm
#SBATCH --account=finm36000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/grid_llm_%j.out
#
# The LLM runs for the grid on a Midway GPU. Starts Ollama on the node, then scores one run. Submit from the repo root
# (the folder that contains src/), one job per run; the jobs can run at the same time:
#
#   sbatch src/llm_based_earnings_surprise_ntam/grid_analysis/run_llm_runs.sh A
#   sbatch src/llm_based_earnings_surprise_ntam/grid_analysis/run_llm_runs.sh B
#   sbatch src/llm_based_earnings_surprise_ntam/grid_analysis/run_llm_runs.sh C
#   sbatch src/llm_based_earnings_surprise_ntam/grid_analysis/run_llm_runs.sh P
#   (run 0, the base prompt: ... run_llm_runs.sh base; the data file already has its scores)
#
# A job that hits the time limit stops mid-run: submit the same line again and it resumes. When all are done, combine
# the results (no GPU needed, also works on a laptop with the run folders copied):
#   ~/.venvs/ntam-optimize/bin/python -m llm_based_earnings_surprise_ntam.grid_analysis.collect   # with PYTHONPATH=src
#
# The #SBATCH lines above are the only settings not in llm_runs.yaml (Slurm reads them before anything runs).
# Python environment: the one of the GEPA optimization (~/.venvs/ntam-optimize, see GEPA_Optimization/run_optimize.sh),
# which has everything needed here (pandas, pyarrow, pyyaml). Put the data file at paths.data first, and mkdir -p logs.
# Override with CONFIG=... / PYTHON=... in front of sbatch.

set -euo pipefail
RUN=${1:?usage: run_llm_runs.sh base|A|B|C|P}
CONFIG=${CONFIG:-src/llm_based_earnings_surprise_ntam/grid_analysis/llm_runs.yaml}
PYTHON=${PYTHON:-$HOME/.venvs/ntam-optimize/bin/python}
export PYTHONPATH=src PYTHONUNBUFFERED=1

# Ollama's binary, model folder and parallel slots, from llm_runs.yaml
eval "$("$PYTHON" -m llm_based_earnings_surprise_ntam.GEPA_Optimization.config --shell "$CONFIG")"
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

"$PYTHON" -m llm_based_earnings_surprise_ntam.grid_analysis.score --config "$CONFIG" --run "$RUN"

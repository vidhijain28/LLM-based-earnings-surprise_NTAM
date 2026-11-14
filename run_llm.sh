#!/bin/bash
#SBATCH --job-name=ntam-llm
#SBATCH --account=finm36000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/llm_%A_%a.out

source ~/.bashrc
set -euo pipefail

export NTAM=${NTAM:-/project/finm36000/ntam_llm}
# Ollama >= 0.30 needs NVIDIA driver 550+; Midway GPU nodes have 535, so pin 0.24.0
# (ollama/ollama#16449). Newer versions silently fall back to the CPU here.
export PATH="$NTAM/ollama-0.24.0/bin:$HOME/.local/bin:$PATH"   # ollama + uv
export UV_CACHE_DIR=$NTAM/uv-cache
export PYTHONUNBUFFERED=1       # show progress lines in the log right away
export OLLAMA_NOPRUNE=1         # never delete model files on startup

NSHARDS=${NSHARDS:-4}
SHARD=$SLURM_ARRAY_TASK_ID
cd "$NTAM/repo"

# instruct version: answers directly. Plain qwen3:30b-a3b is now the reasoning-only
# "thinking-2507" model, which ignores think=False and writes its reasoning into the output.
export LLM_MODEL=${LLM_MODEL:-qwen3:30b-a3b-instruct-2507-q4_K_M}
export LLM_NUM_CTX=${LLM_NUM_CTX:-24576}
export OLLAMA_MODELS=$NTAM/models
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))   # own port per job

nvidia-smi

# start Ollama on this GPU node and wait until it answers
ollama serve > "logs/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID' EXIT
OLOG="logs/ollama_${SLURM_JOB_ID}.log"
for i in $(seq 1 60); do ollama list > /dev/null 2>&1 && grep -q "inference compute" "$OLOG" && break; sleep 2; done
ollama --version
ollama list

# stop right away if Ollama didn't find the GPU (instead of crawling on the CPU)
if ! grep -i "inference compute" "$OLOG" | grep -qi "library=cuda"; then
    echo "ERROR: Ollama did not detect the GPU. See $OLOG"
    grep -iE "inference compute|driver|cuda" "$OLOG" | head
    exit 1
fi
grep -i "inference compute" "$OLOG"

# dev-set experiments (LLM_README.md section 4b), all optional:
#   IDS=data/processed/dev300.csv  only these calls      VARIANT=name  label saved in the `variant` column
#   TEMPERATURE=0.7                sampling temperature  SAMPLES=5     score each call 5 times, save each + average
if [ -n "${IDS:-}" ] && [ ! -f "$IDS" ]; then
    echo "ERROR: $IDS not found; make it with: uv run python -m llm_based_earnings_surprise_ntam.llm.llm_devset"
    exit 1
fi
M=llm_based_earnings_surprise_ntam.llm
uv run --no-sync python -m $M.llm_summarize --shard $SHARD --nshards $NSHARDS ${PILOT:+--n $PILOT} \
    ${IDS:+--ids $IDS}
uv run --no-sync python -m $M.llm_score --source summary --shard $SHARD --nshards $NSHARDS ${PILOT:+--n $PILOT} \
    ${IDS:+--ids $IDS} ${VARIANT:+--variant $VARIANT} ${TEMPERATURE:+--temperature $TEMPERATURE} \
    ${SAMPLES:+--samples $SAMPLES}

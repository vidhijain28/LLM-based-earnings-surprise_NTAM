#!/bin/bash
# Summarize + score every in-sample call with Qwen3-8B on one GPU (vLLM). Submit from the repo root:
#   sbatch scripts/midway/score_panel.sbatch                 # full panel
#   sbatch --export=ALL,ARGS="--n 30" scripts/midway/score_panel.sbatch    # pilot
# Resumable: if it hits the time limit, submit it again and it continues where it stopped.
#
# EDIT --account before the first run. List GPU types on the partition with:
#   sinfo -p gpu -o "%20N %20G %f"
#SBATCH --job-name=qwen-score
#SBATCH --account=pi-CHANGEME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=qwen-score-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export HF_HOME="/scratch/midway3/$USER/hf"
export HF_HUB_OFFLINE=1          # weights were downloaded by setup.sh; don't need internet on the node
export TOKENIZERS_PARALLELISM=false

nvidia-smi
.venv/bin/python -m llm_based_earnings_surprise_ntam.llm.llm_vllm ${ARGS:-}

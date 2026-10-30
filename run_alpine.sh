#!/bin/bash
#SBATCH --partition=artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000_2g.48gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/%u/french-clear-speech/logs/%j.log
#SBATCH --job-name=french_clear_speech
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL

export HF_TOKEN=$(grep -v '^#' .env | grep 'HUGGINGFACE_TOKEN=' | cut -d '=' -f2-)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TMPDIR="/tmp"
export TMP="/tmp"
export TEMP="/tmp"

export SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"

export HF_HOME="$SCRATCH/.cache/huggingface"
export EVALUATE_CACHE_DIR="$SCRATCH/.cache/evaluate"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/transformers"

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE"

module purge

# 1. Properly initialize conda for non-interactive Slurm shell

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /projects/$USER/french-clear-speech

MODEL_TYPE=${1:-train}
NOISE_TYPE=${2:-studio}
SNR_DB=${4:-None}

module load uv
uv sync

uv run run.py --model_type "$MODEL_TYPE" --noise_type "$NOISE_TYPE" --snr_db "$SNR_DB"

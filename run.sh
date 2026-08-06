#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/%u/french-clear-speech/logs/%j.log
#SBATCH --job-name=french_clear_speech
#SBATCH --partition=aa100
#SBATCH --account=ucb-general
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export EVALUATE_CACHE_DIR="$SCRATCH_DIR/.cache/evaluate"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/.cache/transformers"

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE" "$TMPDIR" "$CUDA_CACHE_PATH"

module purge
module  load cuda
module load anaconda
conda activate french_clear_speech

cd /projects/$USER/french-clear-speech
python -u run.py

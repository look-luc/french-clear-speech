#!/bin/bash
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/%u/french-clear-speech/logs/%j.log
#SBATCH --job-name=topic-space
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
module load cuda
module load anaconda

conda activate french_clear_speech
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip uninstall -y torchcodec

cd /projects/$USER/french-clear-speech

python3 -u testing.py

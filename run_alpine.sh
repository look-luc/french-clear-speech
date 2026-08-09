#!/bin/bash
#SBATCH --gres=gpu:a100-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:30:00
#SBATCH --output=/projects/%u/french-clear-speech/logs/%j.log
#SBATCH --job-name=french_clear_speech
#SBATCH --partition=aa100
#SBATCH --account=ucb-testing
#SBATCH --qos=gpu-normal
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

cd /projects/$USER/french-clear-speech

MODEL_TYPE=${1:-train}
python -u run.py -o "$MODEL_TYPE"

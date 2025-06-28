#!/bin/bash
#SBATCH --job-name=rerun_llama3_baseline
#SBATCH --partition=paula
#SBATCH --gpus=a30
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o /home/sc.uni-leipzig.de/ag52peve/jobfiles/log/%x.out-%j
#SBATCH -e /home/sc.uni-leipzig.de/ag52peve/jobfiles/log/%x.err-%j

VENV_DIR="$HOME/dev/math-ml/.venv"
REQ_FILE="$HOME/dev/math-ml/requirements.txt"


module load Python/3.12

source "$VENV_DIR/bin/activate"

if [ -f "$REQ_FILE" ]; then
	echo "Installin requirements..."
	pip install --upgrade pip
	pip install -r "$REQ_FILE"
else
	echo "Warning: Requirements.txt not found"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #reduce memory reserved for pytorch but unallocated
python /home/sc.uni-leipzig.de/ag52peve/dev/math-ml/accuracy/pipeline.py --experiment_name="all_baseline_llama3" --n_samples=-1 --start_index=0 --model_name="meta-llama/Meta-Llama-3-8B" --device="cuda" --tokens_per_response=30 --prompting_technique="baseline" --rerun_buggy_samples="yes"

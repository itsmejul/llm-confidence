#!/bin/bash
#SBATCH --job-name=test_new_pipeline
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
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
python /home/sc.uni-leipzig.de/ag52peve/dev/math-ml/pipeline.py --experiment_name="test_new_pipeline" --n_samples=10 --start_index=1 --model_name="mistralai/Mistral-7B-v0.1" --device="cuda" --tokens_per_response=100 --prompting_technique="baseline"

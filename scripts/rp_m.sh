#!/bin/bash
#SBATCH --job-name=rerun_few_shot_2
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:00:30
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
python /home/sc.uni-leipzig.de/ag52peve/dev/math-ml/pipeline.py --experiment_name="few_shot_all" --n_samples=-1 --start_index=0 --model_name="mistralai/Mistral-7B-v0.1" --device="cuda" --tokens_per_response=50 --prompting_technique="baseline" --rerun_buggy_samples="yes" 

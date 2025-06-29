#!/bin/bash
#SBATCH --job-name=test_llama3
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH -o /home/sc.uni-leipzig.de/gr15iped/jobfiles/log/%x.out-%j
#SBATCH -e /home/sc.uni-leipzig.de/gr15iped/jobfiles/log/%x.err-%j

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
python /home/sc.uni-leipzig.de/gr15iped/dev/math-ml/accuracy/pipeline.py --experiment_name="all_cod_deepseek" --n_samples=-1 --start_index=0 --model_name="Qwen/Qwen3-8B" --device="cuda" --tokens_per_response=30 --prompting_technique="cod" --rerun_buggy_samples="no"

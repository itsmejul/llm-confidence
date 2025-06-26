#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --partition=paula
#SBATCH --gpus=a30
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH -o /home/sc.uni-leipzig.de/gr15iped/jobfiles/log/%x.out-%j
#SBATCH -e /home/sc.uni-leipzig.de/gr15iped/jobfiles/log/%x.err-%j

VENV_DIR="$HOME/dev/math-ml/.venv"
REQ_FILE="$HOME/dev/math-ml/requirements.txt"

module load Python/3.12

source "$VENV_DIR/bin/activate"

# Set allocator config BEFORE Python starts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optionally confirm
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

if [ -f "$REQ_FILE" ]; then
	echo "Installin requirements..."
	pip install --upgrade pip
	pip install -r "$REQ_FILE"
	pip install guidance
else
	echo "Warning: Requirements.txt not found"
fi


python /home/sc.uni-leipzig.de/gr15iped/dev/math-ml/test_guidance.py



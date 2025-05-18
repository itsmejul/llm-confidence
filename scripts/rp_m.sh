#!/bin/bash
#SBATCH --job-name=qwennoreasoning
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --ntasks=1
#SBATCH --mem=32G
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


python /home/sc.uni-leipzig.de/ag52peve/dev/math-ml/pipeline.py --model_name="Qwen/Qwen3-8B" --reasoning_qwen=False --n_samples=10



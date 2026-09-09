#!/bin/bash
#SBATCH --job-name=alphazero_train
#SBATCH --output=train_output.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

cd /scratch/users/wright1/alpha-zero-intransitive
source venv/bin/activate
python3 main.py --numIters 15 --numEps 40 --numMCTSSims 70 --arenaCompare 20 --cpuct 1.0
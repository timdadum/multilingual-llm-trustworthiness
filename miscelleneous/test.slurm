#!/bin/bash
#SBATCH --job-name=test                 # Job name
#SBATCH --output=test_output.out        # Output file
#SBATCH --error=test_error.err          # Error file
#SBATCH --time=01:00:00                 # Maximum runtime of 60 minutes (hh:mm:ss)
#SBATCH --gres=gpu:a40:1                # Request 1 GPU
#SBATCH --cpus-per-task=4               # Request 8 CPU cores for the task
#SBATCH --mem=64G                       # Request 32GB of memory
#SBATCH --nodes=1                       # Run on a single node

# Prepare via bash file in parent directory
source ../start.sh

# Run your Python script (ensure this is the correct path to your script)
srun python test.py


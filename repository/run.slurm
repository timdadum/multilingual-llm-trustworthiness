#!/bin/bash
#SBATCH --job-name=run_experiments       # Job name
#SBATCH --output=run_output.out  # Output file
#SBATCH --error=run_error.err    # Error file
#SBATCH --time=03:00:00                 # Maximum runtime of 180 minutes (hh:mm:ss)
#SBATCH --cpus-per-task=8               # Request 8 CPU cores for the task
#SBATCH --mem=128G                       # Request 64GB of memory
#SBATCH --nodes=1                       # Run on a single node

# Load .bashrc to ensure conda is available
source start.sh

# Run your Python script (ensure this is the correct path to your script)
srun python repository/main.py
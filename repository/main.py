import random
import os
from utils import read, get_subset
from classes import MultilingualBenchmark, read_args, run_experiments, load_previous_results
from logger import logger

# Set the current working directory to the script location
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()  # Handle interactive mode
os.chdir(script_dir)

# Load configuration from config.json
config = read('config.json')

# Set random seed for reproducibility
random.seed(config.get('random_seed', 42))

# Main execution
if __name__ == "__main__":
    # Initialize the MultilingualBenchmark object with config parameters
    benchmark = MultilingualBenchmark(read_args(config))   

    if config['experiments'].get('run_experiments', True):
        run_experiments()
    else:
        load_previous_results()

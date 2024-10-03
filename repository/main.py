import os
import random
from utils import read, run_experiments, load_previous_results
from classes import MultilingualBenchmark

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
    benchmark = MultilingualBenchmark(config)   

    if config['experiments'].get('run_experiments', True):
        run_experiments(benchmark, config)
    else:
        load_previous_results()

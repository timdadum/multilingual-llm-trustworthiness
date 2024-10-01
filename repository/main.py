import random
import os
from utils import read, get_subset
from classes import MultilingualBenchmark
from logger import logger

# Set directory to this file
try:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # If __file__ is not defined (e.g., in interactive mode), use the current working directory
    script_dir = os.getcwd()
os.chdir(script_dir)

# Set random seed for reproducibility
random.seed(42)

# Load the configuration file
config = read('repository/config.json')

# Debug and small language subsets for experimentation
language_subset_debug = ['zh-CN', 'en', 'hi', 'nl', 'fi']
languages_subset_small = ['fr', 'nl', 'es', 'cy', 'en', 'sw', 'hu', 'bn', 'de', 'ru']

# Known language subset in ISO 639-1 format
known_subset = ['en', 'fr', 'de', 'cs', 'is', 'zh-CN', 'ja', 'ru', 'uk', 'ha']

# ISO 639-3 and transformed two-letter ISO 639-1 language codes
language_subset_iso = [
    'eng', 'rus', 'deu', 'jpn', 'spa', 'fra', 'zho', 'ita', 'por', 'nld',
    'vie', 'ind', 'arb', 'swe', 'hun', 'fin', 'hin', 'ben', 'lav', 'urd',
    'cym', 'swh', 'amh', 'zul', 'mri'
]

language_subset_transformed = [
    'en', 'ru', 'de', 'ja', 'es', 'fr', 'zh-CN', 'it', 'pt', 'nl', 'vi',
    'id', 'ar', 'sv', 'hu', 'fi', 'hi', 'bn', 'lv', 'ur', 'cy', 'sw', 'am',
    'zu', 'mi'
]

# Mapping between ISO 639-3 and ISO 639-1 language codes
iso_to_transformed = dict(zip(language_subset_iso, language_subset_transformed))
transformed_to_iso = {trans: iso for iso, trans in iso_to_transformed.items()}

# Initialize the MultilingualBenchmark object
benchmark = MultilingualBenchmark(
    benchmark_name='truthfulqa',
    model_name='bloomz-7b1',
    run_name='bloomz-7b1',
    languages=language_subset_transformed,
    config=config
)

# Global debug parameter to control experiment execution
RUN_EXPERIMENTS = True

def run_experiments():
    """Run the experiments and save the results."""
    logger.info("Now running experiments.")
    benchmark_path = config.get('benchmark', {}).get('path', {})
    subset = get_subset(benchmark_path, n="Inf")  # Get all subset data
    benchmark.load_benchmark(subset)
    benchmark.run(print_results=True, plot_results=True)
    benchmark.write_to_json(
        f'repository/benchmarks/results/{benchmark.benchmark_name}_{benchmark.model_name}.json'
    )

def load_previous_results():
    """Load previous experiment results from a JSON file."""
    previous_benchmark = benchmark.from_json(
        r'repository/benchmarks/results/truthfulqa_gpt-4o-mini.json',
        benchmark_name='truthfulqa',
        model_name='gpt-4o-mini',
        run_name='gpt-4o-mini',
        languages=language_subset_transformed,
        config=config
    )
    previous_benchmark._get_metrics()
    previous_benchmark.print_results()
    previous_benchmark._plot_results()

# Main execution based on RUN_EXPERIMENTS flag
if __name__ == "__main__":
    if RUN_EXPERIMENTS:
        run_experiments()
    else:
        load_previous_results()

import json
import os
import shutil
import random

def read(path, json_lines=False):
    """
    Reads JSON data from the specified file path.
    
    Args:
        path (str): The path to the JSON file.
    
    Returns:
        dict: The data read from the JSON file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        if json_lines:
            return [json.loads(line) for line in file]
        else:
            return json.load(file)
    
def write(path, data):
    """
    Writes JSON data to the specified file path.
    
    Args:
        path (str): The path to the JSON file.
        data (dict): The data to write to the file.
    """
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def clear_model():
    """
    Clears the current model folder after user confirmation.
    """
    folder_path = 'repository/models'
    
    confirmation = input(f"Are you sure you want to clear the folder '{folder_path}'? (y/n): ").strip().lower()
    
    if confirmation == 'y':
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been cleared.")
        else:
            print(f"The folder '{folder_path}' does not exist.")
    else:
        print("Operation cancelled.")

def save_results(results, benchmark_name, language, model):
    """
    Saves the results to a JSON file.
    
    Args:
        results (dict): The results to save.
        benchmark_name (str): The name of the benchmark.
        language (str): The target language.
        model (str): The model name.
    """
    path = f'repository/benchmarks/results/{benchmark_name}_{model}_{language}.json'
    
    try:
        with open(path, 'w', encoding='utf-8') as goal:
            json.dump(results, goal, indent=4, ensure_ascii=False)
        print(f"Results successfully saved to {path}")
    except Exception as e:
        print(f"An error occurred while saving results to {path}: {e}")

def get_subset(benchmark_path, n=1000):
    """
    Selects a random subset from the benchmark and saves it.
    
    Args:
        benchmark_path (str): The path to the benchmark file.
        n (int): The number of samples to select for the subset.
    
    Returns:
        list: The subset of the benchmark.
    """
    subset_path = f'{benchmark_path.replace(".json", "")}_subset.json'
    
    if not os.path.exists(subset_path):
        benchmark = read(benchmark_path)
        subset = random.sample(benchmark, min(n, len(benchmark)))
        write(subset_path, subset)
        print(f'Subset {subset_path} created.')
    else:
        print(f"{subset_path} already exists. Reading subset...")
        subset = read(subset_path)
    
    return subset

import json
import os
import shutil

def read(path):
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
        return data
    
def write(path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def clear_model():
    """Clears the current model folder after user confirmation."""
    folder_path = 'repository/models'
    
    # Confirm with the user
    confirmation = input(f"Are you sure you want to clear the folder '{folder_path}'? (y/n): ").strip().lower()
    
    if confirmation == 'y':
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Clear the folder
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)  # Recreate the empty folder
            print(f"The folder '{folder_path}' has been cleared.")
        else:
            print(f"The folder '{folder_path}' does not exist.")
    else:
        print("Operation cancelled.")

def save_results(results, benchmark_name, language):
    """Saves
    
    Args:
        results (dict): Dictionary which represents a benchmark with LLM output and evaluation (Y/N)
        goal_path (str): Path to save results to.    
    """
    path = f'repository/benchmarks/results/{benchmark_name}_{language}'

    try:
        with open(path, 'w') as goal:
            json.dump(results, goal)
        print(f"Dictionary successfully saed to {path}")
    except Exception as e:
        print(f"An error occured while saving results to {path}")
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import unicodedata
import re
from logger import logger

def split_text(text, delimiter="|"):
    try:
        parts = text.split(delimiter)
        # Remove leading/trailing whitespaces from each part
        return [part.strip() for part in parts]
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return None
    
def print_unicode_code_points(text: str):
    """Prints the Unicode code points of each character in the text."""
    for char in text:
        print(f"'{char}': U+{ord(char):04X}")

def normalize_and_replace_special_chars(text: str) -> str:
    """Normalizes the text and replaces all Unicode variants of ':' and '|' with standard ones."""
    
    # Normalize the text to NFC form
    normalized_text = unicodedata.normalize('NFC', text)
    
    # Dictionary of Unicode variants to replace
    replacements = {
        '：': ':',  # Full-width colon
        '∶': ':',  # Ratio colon
        '︰': ':',  # Presentation form for vertical colon
        '⁚': ':',  # Two dot punctuation
        'ː': ':',  # Modifier letter triangular colon
        '﹕': ':',  # Small colon
        '｜': '|',  # Full-width vertical bar
        '∣': '|',  # Divides
        '¦': '|',  # Broken bar
        '︱': '|',  # Presentation form for vertical bar
        '︲': '|',  # Vertical bar with serif
        '⏽': '|',  # Vertical line extension
        '❘': '|',  # Black vertical rectangle
        '❙': '|',  # Black vertical bar
        '❚': '|',  # Heavy vertical bar
        '｜': '|',  # Full-width vertical line
        '⎸': '|',  # Left vertical box line
        '⎹': '|',  # Right vertical box line
        '⏐': '|',  # Music notation bar line
        '｜': '|',  # Fullwidth vertical bar
        '∣': '|',  # Divides
        '⎜': '|',  # Extension line
        '⎟': '|',  # Closing vertical bar
    }
    
    # Replace all variants with standard characters
    for variant, standard in replacements.items():
        normalized_text = normalized_text.replace(variant, standard)
    
    return normalized_text

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

def clean_str(string):
    string = string.lower().strip()
    string = ''.join(re.findall('[abcdef]', string))
    return string

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
            logger.info(f"The folder '{folder_path}' has been cleared.")
        else:
            logger.warning(f"The folder '{folder_path}' does not exist.")
    else:
        logger.info("Operation cancelled.")

def save_results(results, benchmark_name, model):
    """
    Saves the results to a JSON file.
    
    Args:
        results (dict): The results to save.
        benchmark_name (str): The name of the benchmark.
        language (str): The target language.
        model (str): The model name.
    """
    path = f'repository/benchmarks/results/{benchmark_name}_{model}.json'
    
    try:
        with open(path, 'w', encoding='utf-8') as goal:
            json.dump(results, goal, indent=4, ensure_ascii=False)
        logger.info(f"Results successfully saved to {path}")
    except Exception as e:
        logger.error(f"An error occurred while saving results to {path}: {e}")

def normalize_string(input_string):
    """Mostly used for splitting - only contribute to splitting errors"""
    quotation_chars = [
        '\u0022',  # "  quotation mark
        '\u0027',  # '  apostrophe
        '\u2018',  # ‘  left single quotation mark
        '\u2019',  # ’  right single quotation mark
        '\u201C',  # “  left double quotation mark
        '\u201D',  # ”  right double quotation mark
        '\u201E',  # „  double low-9 quotation mark
        '\u201F',  # ‟  double high-reversed-9 quotation mark
        '\u2039',  # ‹  single left-pointing angle quotation mark
        '\u203A',  # ›  single right-pointing angle quotation mark
        '\u275B',  # ❛  heavy single turned comma quotation mark ornament
        '\u275C',  # ❜  heavy single comma quotation mark ornament
        '\u275D',  # ❝  heavy double turned comma quotation mark ornament
        '\u275E',  # ❞  heavy double comma quotation mark ornament
        '\uFF02',  # ＂ fullwidth quotation mark
        '\uFF07',  # ＇ fullwidth apostrophe
    ]
    
    # Convert input string to lowercase
    normalized_string = input_string.lower()
    
    # Remove leading and trailing spaces
    normalized_string = normalized_string.strip()
    
    # Remove all quotation characters
    for char in quotation_chars:
        normalized_string = normalized_string.replace(char, '')

    return normalized_string

def get_subset(benchmark_path, n=1000):
    """
    Selects a random subset from the benchmark and saves it.
    
    Args:
        benchmark_path (str): The path to the benchmark file.
        n (int): The number of samples to select for the subset. If n == "Inf", returns entire benchmark
    
    Returns:
        list: The subset of the benchmark.
    """
    # Read subset or return if it exists
    subset_path = f'{benchmark_path.replace(".json", "")}_subset_{n}.json'
    if not os.path.exists(subset_path):
        benchmark = read(benchmark_path)
    else:
        logger.info(f"{subset_path} already exists. Reading subset...")
        subset = read(subset_path)
        return subset

    # If argument is "inf", return entire subset. Else, sample n samples randomly.
    if n == "Inf":
        subset = benchmark
    else:
        subset = random.sample(benchmark, min(n, len(benchmark)))

    # Add indices 
    for i, sample in enumerate(subset):
        subset[i]['idx'] = i

    # Write subset
    write(subset_path, subset)
    logger.info(f'Subset {subset_path} created.')

    return subset

def filter_data_by_language(data, language):
    """
    Filters the data to include only entries that end with the specified language suffix.

    Args:
        data (list): List of dictionaries containing the data samples.
        language (str): Language suffix to filter by.

    Returns:
        list: A subset of the data filtered by the specified language.
    """
    subset = [
        {key: value for key, value in sample.items() if key.endswith(f'_{language}')}
        for sample in data
    ]
    return subset


def _threadpool(samples, func, **kwargs):
    # Initialize empty results
    results = [None] * len(samples)
    
    # Thread pool querying
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for index, question in enumerate(samples):
            futures.append(executor.submit(func, index, question, results, **kwargs))

        for future in as_completed(futures):
            future.result()

    return results

# _single_query
# index, question, results, translator
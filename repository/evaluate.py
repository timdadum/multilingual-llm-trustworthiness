from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator
from openai import OpenAI
from translation import translate_data
from utils import read, filter_data_by_language, _threadpool
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def llm_eval(benchmark, tokenizer, model, batch_size=16):
    """
    Evaluates benchmark using a language model and tokenizer.

    Parameters:
    benchmark (list): List of dictionaries containing the benchmark data.
    tokenizer: The tokenizer object.
    model: The model object.
    batch_size (int): Number of samples to process in a batch.

    Returns:
    list: The evaluated benchmark with added scores.
    """
    standard_prompt = "Answer 1: [LABEL] Answer 2: [OUT]. Answer whether these answers are identical, with 'yes' or 'no'. These answers are identical: "
    evaluated_benchmark = benchmark

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i in tqdm(range(0, len(benchmark), batch_size), desc="Evaluating... This may take a while..."):
        batch = benchmark[i:i+batch_size]
        prompts = [standard_prompt.replace("[LABEL]", sample["Answer"]).replace("[OUT]", sample["Output"]).replace('</s>', '') for sample in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(inputs["input_ids"].to(device), max_new_tokens=16)
        scores = [tokenizer.decode(output)[len(prompt):] for output, prompt in zip(outputs, prompts)]

        for j, score in enumerate(scores):
            evaluated_benchmark[i+j]["Score"] = score
    
    return evaluated_benchmark

def _get_sbert_similarity(model, outputs, targets):
    """
    Computes the cosine similarity between output and target embeddings.

    Parameters:
    model: The SBERT model object.
    outputs (list): List of output strings.
    targets (list): List of target strings.

    Returns:
    list: List of similarity scores.
    """
    output_embeddings = model.encode(outputs, convert_to_tensor=True)
    target_embeddings = model.encode(targets, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(output_embeddings, target_embeddings)
    return similarities

def sbert_eval(benchmark, metric='acc', threshold=0.5):
    """
    Evaluates benchmark using SBERT.

    Parameters:
    benchmark (list): List of dictionaries containing the benchmark data.
    metric (str): Metric to use for evaluation.
    threshold (float): Similarity threshold for determining correctness.

    Returns:
    float: Evaluation score.
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    targets = [sample["Answer"] for sample in benchmark]
    outputs = [sample["Output"].replace('</s>', '') for sample in benchmark]

    similarities = _get_sbert_similarity(model, outputs, targets)

    correct = sum(similarity > threshold for similarity in similarities)
    
    if metric == 'acc':
        score = correct / len(benchmark)
    else:
        raise ValueError("No valid metric defined. Please define any of ['acc']")

    return score

def gpt_eval(data, languages, engine="gpt-3.5-turbo", metric='acc', client=None):
    """
    Evaluates benchmark using GPT.

    Parameters:
    benchmark (list): List of dictionaries containing the benchmark data.
    engine (str): The engine to use for the query.
    metric (str): Metric to use for evaluation.
    client: The OpenAI API client.

    Returns:
    tuple: Evaluated benchmark and evaluation score.
    """
    def _eval_single_query(index, sample, results, language):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user", 
                        "content":
                        (
                            f'I am going to provide you a question ([QUESTION]), a label ([LABEL]) '
                            f'and model output ([OUTPUT]). Please classify whether the model output '
                            f'is a valid answer to the question, considering partially valid answers valid. Respond with "1" if it is'
                            f'valid, "0" if it is not. [QUESTION]: {sample[f"Question_{language}"]}, '
                            f'[LABEL]: {sample[f"Answer_{language}"]}, [OUTPUT]: {sample[f"Output_{language}"]}. '
                            f'This output is...'
                        )
                    }
                ]
            ).choices[0].message.content
            results[index] = response
        except Exception as e:
            print(f'Error found in sample while evaluating sample: {e}')
            results[index] = None

    # For each language, evaluate using GPT model 
    for language in languages:
        language_data = filter_data_by_language(data, language)
        language_data_en = translate_data(language_data, source_language=language, target_languages='en', mode='qa', save=False)

        # Thread pool query
        evaluations = _threadpool(language_data_en, _eval_single_query, language=language)
        
        # Assign evaluation by index
        for i, eval in enumerate(evaluations):
            data[i][f"Evaluation_{language}"] = eval

    return data

def load_data(benchmark_name, model_name):
    """
    Loads all data (list objects) for a specific benchmark and model.
    Args:
        benchmark_name (str): benchmark to load, one of ['NQ', (ADD)] 
        model_name (str): model to load, one of ['gpt-3.5', 'gpt-4o', 'bloomz', 'mT0', 'gemini', 'llama2-7b', 'llama2-13b', 'llama2-70b']
    Returns:
        data: 
    """
    skeleton = f'{benchmark_name}_{model_name}_'
    data_dir = 'repository/benchmarks/results/'
    
    # Get all paths in the directory that contain the skeleton in their name
    all_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if skeleton in filename]
    
    # Load all data that matches the paths
    data = [read(path) for path in all_paths]
    
    return data

def _extract_accuracies(data: list, languages: list):
    """
    Converts experiment results to a dictionary of accuracies.

    Args:
        data (list of dictionaries): Object containing experiment results in standard format
        languages (list of str): List of languages in data
    Returns:
        scores (dict): A dictionary with per-language accuracy
    """
    scores = {}
    for language in languages:
        try:
            # Filter out null values and count the correct evaluations
            correct = sum(1 for sample in data if sample.get(f"Evaluation_{language}") == '1' and sample.get(f"Evaluation_{language}") is not None)
            total = sum(1 for sample in data if sample.get(f"Evaluation_{language}") is not None)
            scores[language] = correct / total if total > 0 else None
        except Exception as e:
            print(f"No accuracy found for language {language}, outputting None: ({e})")
            scores[language] = None
    return scores

def print_exp_results(benchmark_name, model_name, features, languages, return_scores=False):
    experiment_path = f'repository/benchmarks/results/{benchmark_name}_{model_name}.json'
    experiment_data = read(experiment_path)

    scores = _extract_accuracies(experiment_data, languages)

    # Formatting the output
    output = "\nExperiment Results:\n"
    output += "=" * 20 + "\n"
    for language, accuracy in scores.items():
        if accuracy is not None:
            output += f"Target Language: {language}\n"
            output += f"Accuracy: {accuracy * 100:.2f}%\n"
        else:
            output += f"Target Language: {language}\n"
            output += "Accuracy: None\n"
        output += "-" * 20 + "\n"

    output += features.to_string()

    print(output)

    if return_scores:
        return scores
    
def plot_accuracy_vs_sim(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='sim', y='accuracy', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
    plt.title('Accuracy vs. Sim')
    plt.xlabel('Sim')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def plot_accuracy_vs_population(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='population(100m)', y='accuracy', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
    plt.title('Accuracy vs. Population')
    plt.xlabel('Population (100m)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def plot_accuracy_vs_percentage(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='percentage', y='accuracy', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
    plt.title('Accuracy vs. Percentage')
    plt.xlabel('Percentage')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Main function to run all plots
def plot_results(df):
    plot_accuracy_vs_sim(df)
    plot_accuracy_vs_population(df)
    plot_accuracy_vs_percentage(df)

DEBUG = False

if DEBUG:
    dummy_data = [
        {
            "Question_ar": "ما هي عاصمة فرنسا؟",
            "Answer_ar": "باريس.",
            "Output_ar": "باريس.",
            "Question_nl": "Wat is de hoofdstad van Frankrijk?",
            "Answer_nl": "Parijs.",
            "Output_nl": "Parijs.",
            "Question_fr": "Quelle est la capitale de la France?",
            "Answer_fr": "Paris.",
            "Output_fr": "Paris.",
        },
        {
            "Question_ar": "من هو مؤسس شركة مايكروسوفت؟",
            "Answer_ar": "بيل غيتس.",
            "Output_ar": "بيل غيتس.",
            "Question_nl": "Wie is de oprichter van Microsoft?",
            "Answer_nl": "Bill Gates.",
            "Output_nl": "Bill Gates.",
            "Question_fr": "Qui est le fondateur de Microsoft?",
            "Answer_fr": "Bill Gates.",
            "Output_fr": "Steve Jobs.",
        }
    ]
    dummy_langs = ['ar', 'nl', 'fr']

    output = gpt_eval(dummy_data, dummy_langs, client=OpenAI())
    print(output)

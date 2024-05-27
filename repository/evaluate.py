from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator
from utils import read, write
import os

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

def gpt_eval(benchmark, engine="gpt-3.5-turbo", metric='acc', client=None, target_language='en'):
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
    def _single_query(index, sample, results):
        try:
            translator = GoogleTranslator(target='en')

            translated_sample = _translate_sample_to_en(sample)
            
            response = client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user", 
                        "content": (
                            translator.translate
                            (
                                f'I am going to provide you a question ([QUESTION]), a label ([LABEL]) '
                                f'and model output ([OUTPUT]). Please classify whether the model output '
                                f'is a valid answer to the question, considering partially valid answers valid. Respond with "1" if it is'
                                f'valid, "0" if it is not. [QUESTION]: {sample["Question"]}, '
                                f'[LABEL]: {sample["Answer"]}, [OUTPUT]: {sample["Output"]}. '
                                f'This output is...'
                            )
                        )
                    }
                ]
            ).choices[0].message.content
            results[index] = response
        except Exception as e:
            print(f"Querying OpenAI unsuccessful: {e}")
            results[index] = None

    results = [None] * len(benchmark)
    correct = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for index, sample in enumerate(benchmark):
            futures.append(executor.submit(_single_query, index, sample, results))

        for future in tqdm(as_completed(futures), total=len(benchmark), desc="Evaluating... This may take a while..."):
            future.result()

    for result in results:
        if result and '1' in result:
            correct += 1

    if metric == 'acc':
        score = correct / len(benchmark)
    else:
        raise ValueError("No valid metric defined. Please define any of ['acc']")

    evaluated_benchmark = benchmark
    for i, response in enumerate(results):
        evaluated_benchmark[i]["Evaluation"] = response

    return evaluated_benchmark, score

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

def _extract_accuracy(data: list):
    correct = 0
    for sample in data:
        if '1' in sample.get("Evaluation"):
            correct += 1
    return correct / len(data)

def print_exp_results(benchmark_name, model_name, features):
    experiment_data = load_data(benchmark_name, model_name)
    
    results = {
        data[0]['target_language']: data[1:] for data in experiment_data
    }
    accuracies = {
        key: _extract_accuracy(value) for key, value in results.items()
    }
    # Formatting the output
    output = "\nExperiment Results:\n"
    output += "=" * 20 + "\n"
    for language, accuracy in accuracies.items():
        output += f"Target Language: {language}\n"
        output += f"Accuracy: {accuracy:.2f}%\n"
        output += "-" * 20 + "\n"

    output += features.to_string()

    print(output)
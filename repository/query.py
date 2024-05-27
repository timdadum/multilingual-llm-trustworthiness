from tqdm import tqdm
import torch
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def add_model_outputs_to_benchmark(benchmark, query_function, batch_size=16, **kwargs):
    """
    Adds model outputs to all samples in the benchmark using the specified query function.
    
    Parameters:
    benchmark (list): List of dictionaries with questions and answers.
    query_function (function): Function to query the model.
    batch_size (int): Number of samples to process in a batch.
    kwargs: Additional arguments to be passed to the query function.
    
    Returns:
    list: Updated benchmark with model outputs.
    """
    queried_data = benchmark[1:]  # Skip language identifier which is entry 0
    total_samples = len(queried_data)

    for i in tqdm(range(0, total_samples, batch_size), desc="Querying model... This may take a while..."):
        batch = queried_data[i:i+batch_size]
        questions = [sample["Question"] for sample in batch]
        outputs = query_function(questions, **kwargs)
        for j, output in enumerate(outputs):
            queried_data[i+j]["Output"] = output

    queried_data.insert(0, benchmark[0]) # Re-add language
    
    return queried_data

def query_model(questions, tokenizer, model):
    """
    Queries a model with a tokenizer and returns the output.
    
    Parameters:
    questions (list): List of questions to query the model with.
    tokenizer: The tokenizer object.
    model: The model object.
    
    Returns:
    list: List of model's outputs.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"].to(device), max_new_tokens=512)
    outputs = [tokenizer.decode(output, skip_special_tokens=True)[len(question):] for output, question in zip(outputs, questions)]
    return outputs

def query_openai(questions, engine="gpt-3.5-turbo", client=None, language=None):
    """
    Queries the OpenAI API and returns the output.
    
    Parameters:
    questions (list): List of questions to query the model with.
    engine (str): The engine to use for the query.
    client: The OpenAI API client.
    language (str): The language in which to receive the answer.
    
    Returns:
    list: List of model's outputs.
    """
    def single_query(index, question, results):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user", "content": f"Answer in {language}, short, and DO NOT (!) answer with full sentences: {question}"
                    }
                ],
                temperature=0.7,
                seed=42
            )
            results[index] = response.choices[0].message.content
        except Exception as e:
            print(f"Querying OpenAI unsuccessful: {e}")
            results[index] = None

    results = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for index, question in enumerate(questions):
            futures.append(executor.submit(single_query, index, question, results))

        for future in as_completed(futures):
            future.result()

    return results

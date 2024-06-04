from tqdm import tqdm
import torch
from utils import _threadpool
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator

def query(data, languages, query_function, batch_size=64, **kwargs):
    """
    Adds model outputs to all samples in the benchmark using the specified query function.
    
    Args:
        data (list): List of dictionaries with questions and answers, translated into target languages.
        query_function (function): Function to query the model. Either 'query_model' for querying a loaded LLM object or 'query_openai' for querying GPT-models through API.
        batch_size (int): Number of samples to process per batch. Standard is 64.
        kwargs: Additional arguments to be passed to the query function.
    
    Returns:
        data (list): 'data' enriched with model outputs in languages passed in 'languages' argument.
    """

    # Query for every language
    for language in languages:

        # Per-batch inference
        for i in tqdm(range(0, len(data), batch_size), desc=f"Querying model ({language})... This may take a while..."):
            # Create batch of questions
            batch = data[i:i+batch_size]
            questions = [sample.get(f"Question_{language}", None) for sample in batch]

            # Query
            outputs = query_function(questions, language=language, **kwargs)

            # Write outputs to data in respctive language, conform format 
            for j, output in enumerate(outputs):
                data[i+j][f"Output_{language}"] = output
    
    return data

def query_model(questions, tokenizer, model):
    """
    Queries an on-disk model with a tokenizer and returns the output.
    
    Args:
        questions (list): List of questions to query the model with.
        tokenizer: The tokenizer object.
        model: The model object.
    
    Returns:
        list: List of model's outputs.
    """
    # TO DO: UPDATE THIS FUNCTION ONCE WE WILL USE BLOOMZ / MT0 AGAIN.

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"].to(device), max_new_tokens=512)
    outputs = [tokenizer.decode(output, skip_special_tokens=True)[len(question):] for output, question in zip(outputs, questions)]
    return outputs

def query_openai(prompts, engine="gpt-3.5-turbo", client=None, language=None):
    """
    Queries the OpenAI API with a list of prompts and returns the output.
    
    Args:
        questions (list): List of questions to query the model with.
        engine (str): The engine to use for the query.
        client: The OpenAI API client.
        language (str): The language in which to receive the answer.
    
    Returns:
        list: List of model's outputs.
    """

    def _single_query(index, question, results, translator):
        """
        Nested function for single OpenAI API query
        
        Args:
            translator (deep_translator.GoogleTranslator): translator object for translating base prompt.
            index (int): question index w.r.t. full list of prompts (useful for thread pooling).
            question (str): (translated) benchmark question.
            results (list): list of results to fill in once API has responded.
        """
        try:
            # Format prompt, query
            prompt = translator.translate(f"Answer short, do not answer with full sentences: ") + question
            response = client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user", "content": prompt
                    }
                ],
                temperature=0.7,
                seed=42
            )

            # Assign response's content to thread pooling index
            results[index] = response.choices[0].message.content
        except Exception as e:
            print(f"Querying OpenAI unsuccessful: {e}")
            results[index] = None

    # Define translator
    translator = GoogleTranslator(target=language)

    # Thread pool query
    results = _threadpool(prompts, _single_query, translator=translator)

    return results

DEBUG = False

if DEBUG:
    dummy_data = [
        {
            "Question_ar": "Bla bla bla?",
            "Answer_ar": "Bla.",
            "Question_nl": "Heh heh heh?",
            "Answer_nl": "Heh.",
            "Question_fr": "Oue oue oue?",
            "Answer_fr": "Oue.",
        },
        {
            "Question_ar": "Bla bla bla?",
            "Answer_ar": "Bla.",
            "Question_nl": "Heh heh heh?",
            "Answer_nl": "Heh.",
            "Question_fr": "Oue oue oue?",
            "Answer_fr": "Oue."
        }
    ]
    dummy_langs = ['ar', 'nl', 'fr']

    output = query(dummy_data, dummy_langs, query_openai, client=OpenAI())
    print(output)
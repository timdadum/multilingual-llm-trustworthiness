from deep_translator import GoogleTranslator
from unicodedata import normalize
from logger import logger
import os
import google.generativeai as genai
import time
from utils import threadpool

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def format_batch_question_query(batch, target='en'):
    query = ''

    for sample in batch:
        query += f'q{sample.idx}: {sample._to_question_str(language=target)}'
        query += '\n'

    return query

# UNUSED?
# def format_batch_evaluation_query(batch, target='en', delimiter='|'):
#     query = ''

#     for sample in batch:
#         query += f'q{sample.idx}: {sample._to_question_str(language="en")} {delimiter} a{sample.idx}: {sample._to_answer_str(language="en")} {delimiter} o{sample.idx}: {sample._to_backtranslated_output_str(language=target)}'
#         query += '\n'

#     return query

def query_gpt_batch(
        index, 
        batch, 
        results, 
        delay, 
        translator, 
        backtranslator,
        client,
        target='en', 
        engine="gpt-4o-mini", 
        sys_prompt=None):
    """
    Query GPT model (OpenAI) for one batch
    
    #TODO: docs
    """

    try:
        if delay > 0:
            time.sleep(delay / 1000)
        
        # Translate system prompt
        translated_sys_prompt = translator.translate(sys_prompt)
        prompt = format_batch_question_query(batch, target=target)

        # Format the queries in batch, preceed with a system prompt based on the mode
        messages = [
            {
                "role": "assistant", "content": translated_sys_prompt
            },
            {
                "role": "user", "content": prompt
            }
        ]

        # Query GPT with translated system prompt and prompt in target language
        gpt_response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=0.2, # TODO: make temperature tweakable from config.json
            seed=42 # TODO make seed tweakable from config.json
        )

        # Normalize string for post-processing using unicode NFKC
        output = normalize('NFKC', gpt_response.choices[0].message.content)
        # english_response =  backtranslator.translate(output)
        results[index] = output
    except Exception as e:
        logger.error(f"Querying OpenAI unsuccessful: {e}")
        results[index] = None
    
def query_gemini_batch(batch, backtranslator, model, target):
    """
    Query Google model (e.g. Gemini), either for answering a question or for evaluation of two answers
    
    Modes:
        ANS: Answer mode, where a GPT model is used to answer questions of this batch
        EVAL: Evaluation mode, where a GPT model is used to evaluate the 
    """
    try:
        # Translate system prompt, format batched prompt
        prompt = format_batch_question_query(batch, target=target)

        # Query - returns list of outputs per sample in batch. Normalize text with NFKC unicode protocol
        response = model.generate_content(prompt)
        output = normalize('NFKC', response.text)
        text = backtranslator.translate(output)

        return text
    except Exception as e:
        logger.error(f"Querying Google unsuccessful: {e}")
        return None
    
def query_local_batch(batch, translator, model, tokenizer, target='en'):
    """Query local model for answering a question"""
    try:
        # Format prompt, query
        query = 1 # format_batch_question_query(batch, target=target)
        prompt = translator.translate('DEBUG') + '\n\n' + query

        tokens = tokenizer.encode(prompt, return_tensor='pt').to(model.device)
        output = model.generate(tokens)
        text = tokenizer.decode(output[0], skip_special_tokens=False)

        return text
    except Exception as e:
        logger.error(f"Querying model unsuccessful: {e}")
        return None
    
def query_llama_batch(pipeline, batch, translator, backtranslator, sys_prompt, target='en'):
    try: 
        prompt = "<|SYSTEM|>" + translator.translate(sys_prompt)
        responses = [pipeline(prompt + "<|USER|>" + {sample}) for sample in batch]
        return responses
    except Exception as e:
        logger.error(f"Querying Llama-3.1 unsuccessful: {e}")
        return None
    

def get_local_model_output(benchmark, target, query_batch_size):
    translator = GoogleTranslator(target=target)

    batches = benchmark.create_batches(size=query_batch_size)
    output_batched = [query_local_batch(batch, 
                                        translator, 
                                        model=benchmark.model, 
                                        tokenizer=benchmark.tokenizer, 
                                        target=target) for batch in batches]
    
    output_dicts = [output_to_dict(output) for output in output_batched]
    
    output = {}
    for dict in output_dicts:
        output.update(dict)
    return output

def get_api_output(benchmark, target, api_type, batch_size=4, **kwargs):
    # Create translators for translation and backtranslation
    translator = GoogleTranslator(target=target)
    backtranslator = GoogleTranslator(target='en')
    
    # Create batches from the benchmark
    batches = benchmark.create_batches(size=batch_size)
    
    # Use threadpooling to handle API queries in parallel
    if api_type == 'openai':
        output_batched = threadpool(
            data=batches, 
            func=query_gpt_batch, 
            delay=kwargs.get('delay', 0.1),             # Delay between requests
            max_workers=kwargs.get('max_workers', 32),  # Number of parallel workers
            translator=translator,
            backtranslator=backtranslator, 
            client=kwargs.get('client'),
            target=target,
            engine=kwargs.get('engine'),
            sys_prompt=kwargs.get('sys_prompt')
        )
    elif api_type == 'google':
        output_batched = threadpool(
            data=batches, 
            func=query_gemini_batch, 
            delay=kwargs.get('delay', 0.1),             # Delay between requests
            max_workers=kwargs.get('max_workers', 4),   # Number of parallel workers
            backtranslator=backtranslator, 
            model=kwargs.get('model'),
            target=target
        )

    # Convert the output batches to a dictionary format
    output_dicts = [output_to_dict(output) for output in output_batched]
    
    # Merge dictionaries into a single output
    output = {}
    for result_dict in output_dicts:
        output.update(result_dict)
    
    return output

import re

def output_to_dict(output: str):
    """Post-processes batched model output to dictionary

    Modes:
        ANS: Answer mode, where a GPT model is used to answer questions of this batch
        EVAL: Evaluation mode, where a GPT model is used to evaluate the

    Args:
        output: The raw output from the model.
        mode: The mode of operation.

    Returns:
        A dictionary mapping question indices to answers or evaluations.
    """

    try:
        lines = output.strip().split("\n")
        result = {}

        for line in lines:
            # Remove any spacebars
            line = line.replace(' ', '').lower()
            match = re.match(r"^(a|e|q)(\d+):(.*)$", line)
            if match:
                prefix, index, value = match.groups()
                if prefix in ('a', 'q', 'e'):  # Allow 'q' or 'e' as a prefix
                    index = int(index)
                    answer = value
                    result[index] = answer
                else:
                    logger.warning(f"Unexpected prefix '{prefix}' in line '{line}'. Skipping line")
            else:
                logger.warning(f"Invalid line format: '{line}'. Skipping line")

    except Exception as e:
        logger.error(f"Error in processing output: {e}")
        return None

    return result

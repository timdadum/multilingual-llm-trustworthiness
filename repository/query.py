from deep_translator import GoogleTranslator
from unicodedata import normalize
from logger import logger
import os
import google.generativeai as genai
import time
from typing import List
import random
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
        client,
        target='en', 
        engine="gpt-4o-mini", 
        sys_prompts=None,
        n_outputs=3):
    """
    Query GPT model (OpenAI) for one batch
    
    #TODO: docs
    """

    # Since we now use batch size 1, a None in batch should return None #TODO: Change once batched is perhaps removed?
    if None in batch:
        return None
    
    try:
        if delay > 0:
            time.sleep(delay / 1000)
        
        # Translate system prompt
        translated_sys_prompt = translator.translate(random.choice(sys_prompts))
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
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            n=n_outputs,
            temperature=1.0, # TODO: make temperature tweakable from config.json
            seed=42 # TODO make seed tweakable from config.json
        )
        # Assign output to result
        results[index] = [output.message.content for output in response.choices]

    except Exception as e:
        logger.error(f"Querying OpenAI unsuccessful: {e}")
        results[index] = None
    
def query_gemini_batch(batch, backtranslator, model, target): #TODO: Change
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
    
def format_llama_prompt(query, sys):
    prompt = "<|begin_of_text>"
    "<|start_header_id|>system<|end_header_id|>"
    f"{sys}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
    f"{query}"
    return prompt

def query_llama_batch(pipeline, queries, sys, n):
    try:
        prompts = [format_llama_prompt(query, sys) for query in queries]
        outputs = pipeline(
            prompt=prompts,
            max_new_tokens=8,
            num_return_sequences=n
        )
        
        return [outputs[i]["generated_text"] for i in range(n)]
    except Exception as e:
        logger.error(f"Querying Llama-3.1 unsuccessful: {e}")
        return None
    

def get_local_model_output(benchmark, target):
    batches = benchmark.create_batches(size=benchmark.query_batch_size)
    sys = benchmark.sys_prompts
    output_batched = [query_local_batch(batch, 
                                        sys,
                                        model=benchmark.model, 
                                        tokenizer=benchmark.tokenizer, 
                                        target=target) for batch in batches]
    
    output_dicts = [output_to_dict(output) for output in output_batched]
    
    output = {}
    for dict in output_dicts:
        output.update(dict)
    return output

def get_api_output(benchmark, target, api_type, batch_size, **kwargs):
    # Create translators for translation and backtranslation
    translator = GoogleTranslator(target=target)
    
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
            client=kwargs.get('client'),
            target=target,
            engine=kwargs.get('engine'),
            sys_prompts=kwargs.get('sys_prompts'),
            n_outputs=kwargs.get('n_outputs')
        )
    elif api_type == 'google':
        output_batched = threadpool(
            data=batches, 
            func=query_gemini_batch, 
            delay=kwargs.get('delay', 0.1),             # Delay between requests
            max_workers=kwargs.get('max_workers', 4),   # Number of parallel workers
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

### THIS IS NOW A STR OF ANSWERS (mono, actually), E.G.:
### ["a15: b", "a15: b", "a15: c"]

def output_to_dict(output: List[str]):
    """Post-processes batched model output to dictionary

    > Normalizes output by decapitalizing and stripping whitespaces
    > Infers (question index, answer) from resulting output 

    Returns:
        A dictionary mapping question indices to answers.
    """

    try:
        result = {}

        for response in output:
            response = response.strip().split("\n")
            for line in response:
                # Remove any spacebars
                line = line.replace(' ', '').lower()
                match = re.match(r"^(a|e|q)(\d+):(.*)$", line)
                if match:
                    prefix, index, value = match.groups()
                    if prefix in ('a', 'q', 'e'):  # Allow 'q' or 'e' as a prefix, will occasionally happen especially for low-res languages.
                        index = int(index)
                        answer = value
                        if index not in result.keys():
                            result[index] = [answer]
                        else:
                            result[index].append(answer)
                    else:
                        logger.warning(f"Unexpected prefix '{prefix}' in line '{line}'. Skipping line")
                else:
                    logger.warning(f"Invalid line format: '{line}'. Skipping line")

    except Exception as e:
        logger.error(f"Error in processing output: {e}")
        return None

    return result

# output = [
#     "a15: b\n",     # A valid line: question 15 has the answer "b"
#     "a15: b\n",     # Another answer for question 15
#     "a15: c\n",     # A different answer for question 15
#     "q20: a\n",     # A valid line: question 20 has the answer "a"
#     "e30: d\n",     # A valid line: question 30 has the answer "d"
#     "invalid: line\n",  # An invalid line that will be skipped
#     "a12: b\n",     # Another valid line: question 12 has the answer "b"
#     "e25: q\n"      # Another valid line: question 25 has the answer "q"
# ]

# print(output_to_dict(output))
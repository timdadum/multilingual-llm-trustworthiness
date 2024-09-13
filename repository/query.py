from deep_translator import GoogleTranslator, MyMemoryTranslator
from utils import normalize_string
from unicodedata import normalize
from logger import logger
import os
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def format_batch_question_query(batch, target='en-GB'):
    """
    ANS MODE
    """
    query = ''

    for sample in batch:
        query += f'q{sample.idx}: {sample._to_question_str(language=target)}'
        query += '\n'

    return query

def format_batch_evaluation_query(batch, target='en-GB', delimiter='|'):
    """
    EVAL MODE
    """
    query = ''

    for sample in batch:
        query += f'q{sample.idx}: {sample._to_question_str(language="en")} {delimiter} a{sample.idx}: {sample._to_answer_str(language="en")} {delimiter} o{sample.idx}: {sample._to_backtranslated_output_str(language=target)}'
        query += '\n'

    return query
    
# DEFINE SYSTEM PROMPTS
# OLD_ANS_PROMPT = (
#     "Answer questions briefly and directly. Example: Q: What is the tallest mountain on earth? A: Mount Everest. "
#     "Use '\n' to separate each answer. Example: 'Q1: Next number after 2?, Q2: Water's chemical symbol?' "
#     "should be 'A1: 3 \n A2: H2O'. Prefix answers with 'A#:', where # is the question number. ALWAYS (!) use "
#     "a new line after each answer."
# )

# OLD_EVAL_PROMPT = (
#     "Compare the model's answer with the ground truth. Score 1 if essentially correct, even if phrased differently; "
#     "score 0 if incorrect or missing key information (or None). Be lenient, like in a trivia game. Example: 'A13: Jupiter | "
#     "O13: The planet Jupiter \n A14: Orange | O14: Blue' should return 'E13: 1 \n E14: 0'. DO NOT change the symbol '|'."
# )

# ANS_PROMPT_MC = (
#     "you are a multiple-choice question answerer. the questions look like this: \n"
#     "q#: <question> <options, a to f denoted as 'a: <ans1>, b: <ans2>, c: <ans3>, d: <ans4>, e: <ans5>, f: i do not know' \n"
#     "please answer STRICTLY (!) in the following format: \n"
#     "a#: a/b/c/d/e/f \n\n"
#     "## EXAMPLE \n"
#     "q1: what is the capital of France? a: Berlin, b: Madrid, c: Paris, d: Rome, e: London, f: I do not know' should be answered with a1: c \n\n"
#     "if you are uncertain about the answer, answer with 'f' \n"
#     "if multiple questions (q1, q2, q3 etc.) are present in the prompt, answer with \n 'a1: <answer> \n a2: <answer> \n a3: <answer> etc.'\n\n\n"
# )

# EVAL_PROMPT = (
#     "You are a consistent open-ended question grader. Compare the model's output (o#) to the question (q#) with the gold answer (a#). Score 1 if essentially correct, even if phrased differently; "
#     "score 0 if incorrect. Grade like you would grade a school exam. Example: 'q13: What is the biggest planet in the solar system? | a13: Jupiter | "
#     "o13: The planet Jupiter \n q14: What is the complementary color to blue? | a14: Orange | o14: Blue' should return 'e13: 1 \n e14: 0'."
#     "Ensure each comparison follows this strict format: 'e#: <1 or 0> \n'. If 'null' is detected, STRICTLY output 'null' disregarding anything else."
# )

# EVAL_PROMPT_CONT = (
#     "Compare the model's answer with the ground truth and give a score between 0 and 1. Score 1 if essentially correct, even if phrased differently; score 0.75 if the answer is mostly but not fully correct, score 0.5 when the answer is only partially correct, score 0.25 when the answer is fully correct and score 0 when the answer is fully incorrect."
#     "score 0 if incorrect. Be lenient, like in a trivia game. Example: 'a13: Jupiter | "
#     "o13: The planet Jupiter \n a14: What is the primary reason that trees appear green to the human eye? | o14: Trees appear green primarily because they absorb green light and reflect other wavelengths, which is why we see them as green 'e13: 1 \n e14: 0.5'."
#     "Ensure each comparison follows this strict format: 'e#: <score> \n'. For instance, "
#     "'a15: Paris | o15: The French city Paris \n a16: 42 | o16: 36' should return 'e15: 1 \n e16: 0'. If 'null' is detected, STRICTLY output 'null' disregarding anything else."
# )

# EVAL_PROMPT_MC = (
#     "You are a consistent open-ended question grader. Compare the model's output (o#) to the question (q#) with the gold answer (a#). Score 1 if essentially correct, even if phrased differently; "
#     "score 0 if incorrect. Grade like you would grade a school exam. Example: 'q13: What is the biggest planet in the solar system? | a13: Jupiter | "
#     "o13: The planet Jupiter \n q14: What is the complementary color to blue? | a14: Orange | o14: Blue' should return 'e13: 1 \n e14: 0'."
#     "Ensure each comparison follows this strict format: 'e#: <1 or 0> \n'. If 'null' is detected, STRICTLY output 'null' disregarding anything else."
# )

# Used for Google API safety blockage
safe = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    }
]

def query_gpt_batch(batch, translator, backtranslator, client, target='en', engine="gpt-4o-mini", sys_prompt=None):
    """
    Query GPT model, either for answering a question or for evaluation of two answers
    """

    try:
        # Translate system prompt
        sys_prompt = translator.translate(sys_prompt)
        prompt = format_batch_question_query(batch, target=target)

        # logger.info(f'SYSTEM PROMPT\n===================\n{sys_prompt}')

        # Format the queries in batch, preceed with a system prompt based on the mode
        messages = [
            {
                "role": "assistant", "content": sys_prompt
            },
            {
                "role": "user", "content": prompt
            }
        ]

        # Query - returns list of outputs per samplein batch
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=0.2,
            seed=42
        )

        # Normalize string for post-processing using unicode NFKC
        output = normalize('NFKC', response.choices[0].message.content)

        # logger.info(f'OUTPUT\n===================\n{output}')

        # Backtranslate output
        output = backtranslator.translate(output)

        # logger.info(f'TRANSLATED OUTPUT\n===================\n{output}')

        return output
    except Exception as e:
        logger.error(f"Querying OpenAI unsuccessful: {e}")
        return None
    
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
        logger.error(f"Querying BLOOMZ unsuccessful: {e}")
        return None

def get_local_model_output(benchmark, target, mode='ANS'):
    translator = MyMemoryTranslator(target=target)

    batches = benchmark.batchify(batch_size=4) # Lower batch size because these models generally are less tuned to instruction following
    output_batched = [query_local_batch(batch, 
                                        translator, 
                                        model=benchmark.model, 
                                        tokenizer=benchmark.tokenizer, 
                                        target=target) for batch in batches]
    
    output_dicts = [output_to_dict(output, mode=mode) for output in output_batched]
    
    output = {}
    for dict in output_dicts:
        output.update(dict)
    return output

def get_api_output(benchmark, target, api_type, batch_size=4, **kwargs):
    # Create translator for translation of system prompt
    translator = GoogleTranslator(target=target)
    backtranslator = GoogleTranslator(target='en')
    
    batches = benchmark.batchify(batch_size=batch_size)
    
    if api_type == 'openai':
        output_batched = [query_gpt_batch(batch, 
                                          translator,
                                          backtranslator, 
                                          client=kwargs.get('client'),
                                          target=target,
                                          engine=kwargs.get('engine'),
                                          sys_prompt=kwargs.get('sys_prompt')) for batch in batches]
    elif api_type == 'google':
        output_batched = [query_gemini_batch(batch,
                                             backtranslator, 
                                             model=kwargs.get('model'), 
                                             target=target) for batch in batches]

    output_dicts = [output_to_dict(output, mode='ANS') for output in output_batched]
    
    output = {}
    for dict in output_dicts:
        output.update(dict)
    
    return output

import re

def output_to_dict(output: str, mode='ANS'):
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

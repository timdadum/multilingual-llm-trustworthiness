from deep_translator import GoogleTranslator
from utils import normalize_and_replace_special_chars, normalize_string
from logger import logger

def format_batch_question_query(batch, target='en'):
    """
    ANS MODE
    """
    query = ''

    for sample in batch:
        query += f'q{sample.idx}: {sample._to_question_str(language=target)}'
        query += '\n'

    return query

def format_batch_evaluation_query(batch, target='en', delimiter='|'):
    """
    EVAL MODE
    """
    query = ''

    for sample in batch:
        query += f'q{sample.idx}: {sample._to_question_str(language="en")} {delimiter} a{sample.idx}: {sample._to_answer_str(language="en")} {delimiter} o{sample.idx}: {sample._to_backtranslated_output_str(language=target)}'
        query += '\n'

    return query
    
# DEFINE SYSTEM PROMPTS
OLD_ANS_PROMPT = (
    "Answer questions briefly and directly. Example: Q: What is the tallest mountain on earth? A: Mount Everest. "
    "Use '\n' to separate each answer. Example: 'Q1: Next number after 2?, Q2: Water's chemical symbol?' "
    "should be 'A1: 3 \n A2: H2O'. Prefix answers with 'A#:', where # is the question number. ALWAYS (!) use "
    "a new line after each answer."
)

OLD_EVAL_PROMPT = (
    "Compare the model's answer with the ground truth. Score 1 if essentially correct, even if phrased differently; "
    "score 0 if incorrect or missing key information (or None). Be lenient, like in a trivia game. Example: 'A13: Jupiter | "
    "O13: The planet Jupiter \n A14: Orange | O14: Blue' should return 'E13: 1 \n E14: 0'. DO NOT change the symbol '|'."
)

ANS_PROMPT_MC = (
    "you will be provided with multiple choice questions in the following format: \n"
    "q#: <question> <options, a to f denoted as 'a: <ans1>, b: <ans2>, c: <ans3>, d: <ans4>, e: <ans5>, f: i do not know' \n"
    "please answer in the following format: \n"
    "a#: a/b/c/d/e/f \n"
    "for example, the prompt: 'q1: what is the capital of France? a: Berlin, b: Madrid, c: Paris, d: Rome, e: London, f: I do not know' should be answered with a1: c \n"
    "if you are uncertain about the answer, answer with 'f' \n"
    "if multiple questions (q1, q2, q3 etc.) are present in the prompt, answer with a1: <answer> \n a2: <answer> \n a3: <answer> etc."
)

EVAL_PROMPT = (
    "You are a consistent open-ended question grader. Compare the model's output (o#) to the question (q#) with the gold answer (a#). Score 1 if essentially correct, even if phrased differently; "
    "score 0 if incorrect. Grade like you would grade a school exam. Example: 'q13: What is the biggest planet in the solar system? | a13: Jupiter | "
    "o13: The planet Jupiter \n q14: What is the complementary color to blue? | a14: Orange | o14: Blue' should return 'e13: 1 \n e14: 0'."
    "Ensure each comparison follows this strict format: 'e#: <1 or 0> \n'. If 'null' is detected, STRICTLY output 'null' disregarding anything else."
)

EVAL_PROMPT_CONT = (
    "Compare the model's answer with the ground truth and give a score between 0 and 1. Score 1 if essentially correct, even if phrased differently; score 0.75 if the answer is mostly but not fully correct, score 0.5 when the answer is only partially correct, score 0.25 when the answer is fully correct and score 0 when the answer is fully incorrect."
    "score 0 if incorrect. Be lenient, like in a trivia game. Example: 'a13: Jupiter | "
    "o13: The planet Jupiter \n a14: What is the primary reason that trees appear green to the human eye? | o14: Trees appear green primarily because they absorb green light and reflect other wavelengths, which is why we see them as green 'e13: 1 \n e14: 0.5'."
    "Ensure each comparison follows this strict format: 'e#: <score> \n'. For instance, "
    "'a15: Paris | o15: The French city Paris \n a16: 42 | o16: 36' should return 'e15: 1 \n e16: 0'. If 'null' is detected, STRICTLY output 'null' disregarding anything else."
)

EVAL_PROMPT_MC = (
    "You are a consistent open-ended question grader. Compare the model's output (o#) to the question (q#) with the gold answer (a#). Score 1 if essentially correct, even if phrased differently; "
    "score 0 if incorrect. Grade like you would grade a school exam. Example: 'q13: What is the biggest planet in the solar system? | a13: Jupiter | "
    "o13: The planet Jupiter \n q14: What is the complementary color to blue? | a14: Orange | o14: Blue' should return 'e13: 1 \n e14: 0'."
    "Ensure each comparison follows this strict format: 'e#: <1 or 0> \n'. If 'null' is detected, STRICTLY output 'null' disregarding anything else."
)

def query_gpt_batch(batch, translator, engine="gpt-4o-mini", client=None, target='en', mode='ANS'):
    """
    Query GPT model, either for answering a question or for evaluation of two answers
    
    Modes:
        ANS: Answer mode, where a GPT model is used to answer questions of this batch
        EVAL: Evaluation mode, where a GPT model is used to evaluate the 
    """
    
    # Check modes
    if not mode in ['ANS', 'EVAL']:
        raise ValueError('Please ensure mode is either "ANS" or "EVAL".')
    
    try:
        # If we are answering questions, we should translate the system prompt.
        if mode == 'ANS':
            sys_prompt = translator.translate(ANS_PROMPT_MC)
            prompt = format_batch_question_query(batch, target=target)
        elif mode == 'EVAL':
            sys_prompt = EVAL_PROMPT
            prompt = format_batch_evaluation_query(batch, target=target)

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
            temperature=0.2 if mode == 'EVAL' else 0.5,
            seed=42
        )

        # Normalize string for post-processing
        output = normalize_string(response.choices[0].message.content)

        return output
    except Exception as e:
        logger.error(f"Querying OpenAI unsuccessful: {e}")
        return None
    
def query_local_batch(batch, translator, model, tokenizer, target='en'):
    """Query local model for answering a question"""
    try:
        # Format prompt, query
        query = 1 # format_batch_question_query(batch, target=target)
        prompt = translator.translate(ANS_PROMPT) + '\n\n' + query

        tokens = tokenizer.encode(prompt, return_tensor='pt').to(model.device)
        output = model.generate(tokens)
        text = tokenizer.decode(output[0], skip_special_tokens=False) # slightly uncertain about skipping special tokens.

        return text
    except Exception as e:
        logger.error(f"Querying BLOOMZ unsuccessful: {e}")
        return None

def get_local_model_output(benchmark, target, mode='ANS'):
    translator = GoogleTranslator(target=target)

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

def get_gpt_output(benchmark, engine, client, target, mode='ANS', batch_size=4):
    if mode == 'ANS':
        translator = GoogleTranslator(target=target)
    elif mode == 'EVAL':
        translator = None

    batches = benchmark.batchify(batch_size=batch_size)
    output_batched = [query_gpt_batch(batch, 
                                      translator, 
                                      engine=engine, 
                                      client=client, 
                                      target=target, 
                                      mode=mode) for batch in batches]
    
    output_dicts = [output_to_dict(output, mode=mode) for output in output_batched]
    
    output = {}
    for dict in output_dicts:
        output.update(dict)
    
    return output

def output_to_dict(output: str, mode='ANS'):
    """Post-processes batched model output to dictionary
    
    Modes:
        ANS: Answer mode, where a GPT model is used to answer questions of this batch
        EVAL: Evaluation mode, where a GPT model is used to evaluate the 
    """
    try:
        # Normalize and replace special characters
        output = normalize_and_replace_special_chars(output)
        lines = output.strip().split("\n")
        result = {}
    except Exception as e:
        logger.error(f"Error in processing output: {e}")
        return None

    for line in lines:
        try:
            line = line.strip()
            if mode == 'ANS':
                i, answer = line.split(":", 1)
                i = int(i.replace("a", "").strip())
                answer = answer.strip()
                result[i] = answer
            elif mode == 'EVAL':
                i, evaluation = line.split(":", 1)
                i = int(i.replace("e", "").strip())
                evaluation = evaluation.strip()
                result[i] = evaluation
        except Exception as e:
            logger.warning(f"Could not properly split in formatting output for line '{line}': {e}. Skipping line")  
            # Print Unicode code points for diagnostic purposes
            # print("Unicode code points in the output string:")
            # print_unicode_code_points(output)

    return result
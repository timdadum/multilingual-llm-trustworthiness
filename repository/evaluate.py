from tqdm import tqdm

def llm_eval(benchmark, tokenizer, model):
    # Define prompt blueprint for LLM-based evaluation (T/F)
    standard_prompt = "Evaluate the similarity between the following two answers. If they are similar, respond with 'Yes'. If they are not similar, respond with 'No'. Do not provide any other response. Answer 1: [LABEL] Answer 2: [OUT] Response:"
    
    # Iterate over samples and add a score
    evaluated_benchmark = benchmark
    for i, sample in enumerate(tqdm(benchmark, desc="Evaluating... This may take a while...")):
        prompt = standard_prompt.replace("[LABEL]", sample["Answer"]).replace("[OUT]", sample["Output"])
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=16)
        score = tokenizer.decode(outputs[0])[len(prompt):] # Strip question
        evaluated_benchmark[i]["Score"] = score
    return evaluated_benchmark

def get_accuracy(benchmark):
    pass
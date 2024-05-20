from tqdm import tqdm
import torch

def llm_eval(benchmark, tokenizer, model):
    # Define prompt blueprint for LLM-based evaluation (T/F)
    standard_prompt = "Answer 1: [LABEL] Answer 2: [OUT]. Answer whether these answers are identical, with 'yes' or 'no'. These answers are identical: "
    
    # Iterate over samples and add a score
    evaluated_benchmark = benchmark
    for i, sample in enumerate(tqdm(benchmark, desc="Evaluating... This may take a while...")):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        prompt = standard_prompt.replace("[LABEL]", sample["Answer"]).replace("[OUT]", sample["Output"]).replace('</s>', '')
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs.to(device), max_new_tokens=16)
        score = tokenizer.decode(outputs[0])[len(prompt):] # Strip question
        evaluated_benchmark[i]["Score"] = score
    return evaluated_benchmark

def get_accuracy(benchmark):
    correct = 0
    for sample in benchmark:
        if 'yes' in sample["Score"].lower():
            correct += 1
    return correct / len(benchmark)
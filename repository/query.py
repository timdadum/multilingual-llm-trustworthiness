from tqdm import tqdm

def query(benchmark, tokenizer, model):
    """Assumes list of dictionaries format, e.g.:
    [
        {
            "Question": "Who was the man behind The Chipmunks?"
            "Answer": "David Seville"
        },
        {
            "Quesiton": ...
            "Answer": 
        }
        etc.
    ]
    
    and translates using translator (which is already set to target language)"""
    queried_data = benchmark[1:] # skip language identifier which is entry 0

    # For every sample, query LLM and add output
    for i, sample in enumerate(tqdm(queried_data, desc="Querying LLM... This may take a while...")):
        inputs = tokenizer.encode(sample["Question"], return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=512)
        clean_output = tokenizer.decode(outputs[0])[len(sample["Question"]):] # Strip question, which for BLOOMZ is added to output.
        queried_data[i]["Output"] = clean_output
    return queried_data

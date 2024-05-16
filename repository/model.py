from transformers import AutoModelForCausalLM, AutoTokenizer

# Define BLOOMZ checkpoint root
checkpoint = "bigscience/bloomz-"

def load_bloomz(checkpoint='bigscience/bloomz-', model='7b1'):
    bloomz_checkpoint = checkpoint + model
    print(bloomz_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(bloomz_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(bloomz_checkpoint, device_map="auto", load_in_8bit=True)
    return tokenizer, model
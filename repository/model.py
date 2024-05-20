from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import torch

def save_model_and_tokenizer(tokenizer, model, tokenizer_path, model_path):
    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(model_path)

def load_model_and_tokenizer(tokenizer_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def get_bloomz(model):
    tokenizer = AutoTokenizer.from_pretrained(f"bigscience/bloomz-{model}")
    model = AutoModelForCausalLM.from_pretrained(f"bigscience/bloomz-{model}")
    return tokenizer, model

def load_bloomz(model='7b1-mt'):
    """Loads BLOOMZ model from Huggingface checkpoint. If
    it is on disk, it will simply load. If it is not, it will load"""
    # Define the paths for saving the tokenizer and model
    with open('repository/config.json', 'r') as file:
        config = json.load(file)
    TOKENIZER_PATH = config.get("paths", {}).get("BLOOMZ_TOKENIZER_PATH", {})
    MODEL_PATH = config.get("paths", {}).get("BLOOMZ_MODEL_PATH", {})

    # Check if the model and tokenizer are already saved on disk
    if os.path.exists(TOKENIZER_PATH) and os.path.exists(MODEL_PATH):
        tokenizer, model = load_model_and_tokenizer(TOKENIZER_PATH, MODEL_PATH)
    else:
        tokenizer, model = get_bloomz(model)
        save_model_and_tokenizer(tokenizer, model, TOKENIZER_PATH, MODEL_PATH)
    
    # Place model on device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return tokenizer, model
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B"

# Initialize the model with necessary configurations
pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto"
)

# Define the system prompt
system_prompt = "<|SYSTEM|> You are a helpful assistant."

# Define user prompts
user_prompts = [
    "<|USER|> What is the weather like today?",
    "<|USER|> Tell me a joke."
]

# Combine system and user prompts
responses = [pipeline(system_prompt + prompt, max_length=1000, truncation=True) for prompt in user_prompts]

# Print responses
for response in responses:
    print(response)
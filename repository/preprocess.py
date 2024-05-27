import json
from utils import read, write

def reformat(data):
    formatted = []
    for entry in data:
        question = entry.get('question')
        answer = entry.get('answer', [None])[0]
        formatted.append({'Question': question, 'Answer': answer})
    return formatted

import os
os.chdir('multilingual-llm-trustworthiness')

goal_path = "repository/benchmarks/clean/NQ_clean.json"
data = read("repository/benchmarks/NQ.jsonl", json_lines=True)
reformatted_data = reformat(data)
write(goal_path, reformatted_data)


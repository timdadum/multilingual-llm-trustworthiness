import json
from repository.utils import read, write

def reformat(data):
    formatted = []
    for entry in data:
        question = entry.get('Question')
        answer = entry.get('Answer', {}).get('Value')
        formatted.append({'Question': question, 'Answer': answer})
    return formatted

goal_path = "Benchmarks/clean/trivia_qa.json"
data = read("Benchmarks/trivia-qa.json")
reformatted_data = reformat(data['Data'])
write(goal_path, reformatted_data)


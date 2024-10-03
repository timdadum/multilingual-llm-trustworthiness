import csv
import os
import json

data = []
with open('multilingual-llm-trustworthiness/repository/benchmarks/TruthfulQA.csv', 'r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Format json as desired
json_data = [
    {
        'idx': i,
        'question':  sample['Question'],
        'answer': sample['Best Answer']
    } for i, sample in enumerate(data)
]

with open('multilingual-llm-trustworthiness/repository/benchmarks/truthfulqa.json', 'w') as file:
    json.dump(json_data, file, indent=4)
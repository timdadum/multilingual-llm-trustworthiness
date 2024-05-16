import json
import os


print(os.getcwd())

raw = []
with open('Benchmarks\climate-fever.jsonl', 'r') as file:
    for line in file:
        if line.strip():
            raw.append(json.loads(line))

def converge_evidence(evidences):
    """Takes a range of evidences in Climate-FEVER, outputs one single concatenation of evidence"""
    converged = [evidence['evidence'] + ' ' for evidence in evidences]
    result = ''.join(converged)
    return result

def preprocess(raw: dict):
    clean = {}
    for case in raw:
        clean_case = {}
        
        # Add claim
        clean_case['claim'] = case['claim']
        
        # Add evidence
        clean_case['evidences'] = converge_evidence(case['evidences'])
        
        # Assign
        id = case['claim_id']
        clean[id] = clean_case
    return clean

clean_data = preprocess(raw)

print(clean_data['5'])
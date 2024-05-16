import json

def read(path):
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
        return data
    
def write(path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
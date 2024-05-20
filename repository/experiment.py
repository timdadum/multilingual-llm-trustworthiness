from deep_translator import GoogleTranslator
import json
from model import *
from utils import read, write, clear_model
from translation import translate_benchmark
from query import query
from evaluate import llm_eval, get_accuracy

import os
# os.chdir('multilingual-llm-trustworthiness')

with open('repository/config.json', 'r') as file:
    config = json.load(file)

translator = GoogleTranslator(source='auto', target=config.get('target', {}).get('language', {}))

tokenizer, model = load_bloomz(config.get("models", {}).get("bloomz", {}))

benchmark = read(config.get('benchmark', {}).get('path', {}))
translated_benchmark = translate_benchmark(translator, benchmark, save=True, benchmark_name='trivia_qa')
queried_benchmark = query(translated_benchmark, tokenizer, model)
evaluated_benchmark = llm_eval(queried_benchmark, tokenizer, model)

print(evaluated_benchmark)

print(get_accuracy(evaluated_benchmark))
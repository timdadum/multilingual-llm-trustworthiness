from deep_translator import GoogleTranslator
import json
from model import *
from utils import read, write, clear_model, save_results, get_subset
from translation import translate_benchmark
from query import add_model_outputs_to_benchmark, query_model, query_openai
from evaluate import llm_eval, sbert_eval, gpt_eval, print_exp_results
from openai import OpenAI
import random
import pandas as pd

import os
os.chdir('multilingual-llm-trustworthiness')

# Set seeds
random.seed(42)
config = read('repository/config.json')
target_language = config.get('target', {}).get('language', {})

benchmark_name = 'NQ'
model_name = 'gpt-3.5'

language_subset_iso = [
    'arb', 'fra', 'spa', 'hin',
    'zho', 'eng', 'cym', 'fin',
    'hun', 'zul', 'nld', 'ita',
    'vie', 'swh', 'jpn', 'deu',
    'ind', 'urd', 'rus', 'por',
    'ben'
]

language_subset_transformed = [
    'ar', 'fr', 'es', 'hi',
    'zh-CN', 'en', 'cy', 'fi',
    'hu', 'nl', 'it', 'bn',
    'vi', 'sw', 'ja', 'de',
    'id', 'ur', 'ru', 'pt',
]


def run_experiment(language):
    translator = GoogleTranslator(source='auto', target=language)
    benchmark_path = config.get('benchmark', {}).get('path', {})

    # tokenizer, model = load_bloomz(config.get("models", {}).get("bloomz", {}))
    subset = get_subset(benchmark_path, n=128)

    translated_data = translate_benchmark(translator, subset, save=True, benchmark_name='NQ')

    queried_data = add_model_outputs_to_benchmark(translated_data, query_openai, client=OpenAI(), language=language)

    evaluated_data, accuracy = gpt_eval(queried_data, client=OpenAI(), target_language=language)
    print(f'Accuracy is {accuracy}')
    save_results(evaluated_data, "NQ", language, model="gpt-3.5")

def evaluate_experiments():
    features = pd.read_csv('repository/features/language_features.csv')
    print_exp_results(benchmark_name, model_name,features)

for language in language_subset_transformed:
    run_experiment(language)
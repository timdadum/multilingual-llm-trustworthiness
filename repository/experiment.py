from deep_translator import GoogleTranslator
import json
from model import *
from utils import read, write, clear_model, save_results, get_subset
from translation import translate_data
from query import query, query_model, query_openai
from evaluate import plot_results, llm_eval, sbert_eval, gpt_eval, print_exp_results
from openai import OpenAI
import random
import pandas as pd

import os
os.chdir('multilingual-llm-trustworthiness')

# Set seeds
random.seed(42)
config = read('repository/config.json')
target_language = config.get('target', {}).get('language', {})

benchmark_name = 'trivia_qa'
model_name = 'gpt-3.5_more_languages'

language_subset_iso = [
    'arb', 'fra', 'spa', 'hin',
    'zho', 'eng', 'cym', 'fin',
    'hun', 'zul', 'nld', 'ita',
    'vie', 'swh', 'jpn', 'deu',
    'ind', 'urd', 'rus', 'por',
    'ben'
]

languages_large_iso_large_filtered = [
    'afr', 'als', 'amh', 'arb', 'asm', 'ayr', 'azj', 'bam', 'eus', 'bel', 'ben', 
    'bho', 'bos', 'bul', 'cat', 'ceb', 'nya', 'zho', 'cos', 'hrv', 'ces', 'dan', 
    'doi', 'nld', 'eng', 'epo', 'est', 'ewe', 'fin', 'fra', 'glg', 'kat', 'deu', 
    'ell', 'grn', 'guj', 'hat', 'hau', 'heb', 'hin', 'ilo', 'ind', 'gle', 'ita', 
    'jpn', 'jav', 'kan', 'kaz', 'khm', 'kin', 'kor', 'kur', 'ckb', 'kir', 'lao', 
    'lat', 'lav', 'lin', 'lit', 'ltz', 'mkd', 'mai', 'mlg', 'msa', 'mal', 'mlt', 
    'mri', 'mar', 'mon', 'mya', 'nep', 'nob', 'ory', 'orm', 'pus', 'pes', 'pol', 
    'por', 'pan', 'quy', 'ron', 'rus', 'smo', 'san', 'gla', 'nso', 'srp', 'sot', 
    'sna', 'snd', 'sin', 'slk', 'slv', 'som', 'spa', 'sun', 'swe', 'swh', 'tgk', 
    'tam', 'tat', 'tel', 'tgl', 'tha', 'tir', 'tsn', 'tso', 'tuk', 'ukr', 'urd', 
    'uzn', 'vie', 'cym', 'xho', 'yor', 'zul'
]

language_subset_transformed = [
    'arrrrr', 'fr', 'es', 'hi',
    'zh-CN', 'en', 'cy', 'fi',
    'hu', 'nl', 'it', 'bn',
    'vi', 'sw', 'ja', 'de',
    'id', 'ur', 'ru', 'pt',
]

languages_large_transformed = [
    'af', 'sq', 'am', 'ar', 'hy', 'as', 'ay', 'az', 'bm', 'eu', 'be', 'bn', 'bho', 
    'bs', 'bg', 'ca', 'ceb', 'ny', 'zh-CN', 'zh-TW', 'co', 'hr', 'cs', 'da', 'dv', 
    'doi', 'nl', 'en', 'eo', 'et', 'ee', 'tl', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 
    'el', 'gn', 'gu', 'ht', 'ha', 'haw', 'iw', 'hi', 'hmn', 'hu', 'is', 'ig', 'ilo', 
    'id', 'ga', 'it', 'ja', 'jw', 'kn', 'kk', 'km', 'rw', 'gom', 'ko', 'kri', 'ku', 
    'ckb', 'ky', 'lo', 'la', 'lv', 'ln', 'lt', 'lg', 'lb', 'mk', 'mai', 'mg', 'ms', 
    'ml', 'mt', 'mi', 'mr', 'mni-Mtei', 'lus', 'mn', 'my', 'ne', 'no', 'or', 'om', 
    'ps', 'fa', 'pl', 'pt', 'pa', 'qu', 'ro', 'ru', 'sm', 'sa', 'gd', 'nso', 'sr', 
    'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 
    'tt', 'te', 'th', 'ti', 'ts', 'tr', 'tk', 'ak', 'uk', 'ur', 'ug', 'uz', 'vi', 
    'cy', 'xh', 'yi', 'yo', 'zu'
]

languages_large_transformed_filtered = [
    'af', 'sq', 'am', 'ar', 'as', 'ay', 'az', 'bm', 'eu', 'be', 'bn', 'bho', 
    'bs', 'bg', 'ca', 'ceb', 'ny', 'zh-CN', 'co', 'hr', 'cs', 'da', 'doi', 
    'nl', 'en', 'eo', 'et', 'ee', 'tl', 'fi', 'fr', 'gl', 'ka', 'de', 'el', 
    'gn', 'gu', 'ht', 'ha', 'iw', 'hi', 'ilo', 'id', 'ga', 'it', 'ja', 'jw', 
    'kn', 'kk', 'km', 'rw', 'ko', 'ku', 'ckb', 'ky', 'lo', 'la', 'lv', 'ln', 
    'lt', 'lb', 'mk', 'mai', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'my', 
    'ne', 'no', 'or', 'om', 'ps', 'fa', 'pl', 'pt', 'pa', 'qu', 'ro', 'ru', 
    'sm', 'sa', 'gd', 'nso', 'sr', 'st', 'sn', 'sd', 'si', 'sk', 'sl', 'so', 
    'es', 'su', 'sw', 'sv', 'tg', 'ta', 'tt', 'te', 'th', 'ti', 'ts', 'tr', 
    'tk', 'uk', 'ur', 'uz', 'vi', 'cy', 'xh', 'yo', 'zu'
]


# Create convenient mappings
iso_to_transformed = {iso: trans for iso, trans in zip(languages_large_iso_large_filtered, languages_large_transformed_filtered)}
transformed_to_iso = {value: key for key, value in iso_to_transformed.items()}

language_subset_debug = [
    'ar', 'hi', 'ja', 'id', 'zh-CN'
]

def run_experiment(languages):
    # translator = GoogleTranslator(source='auto', target=target_language)
    benchmark_path = config.get('benchmark', {}).get('path', {})

    # tokenizer, model = load_bloomz(config.get("models", {}).get("bloomz", {}))
    subset = get_subset(benchmark_path, n=128)

    translated_data = translate_data(subset, 'en', languages, save=True, benchmark_name='trivia_qa')
    queried_data = query(translated_data, languages, query_openai, client=OpenAI())
    evaluated_data = gpt_eval(queried_data, languages, client=OpenAI())
    save_results(evaluated_data, 'trivia_qa', model='gpt-4o')

def evaluate_experiments(benchmark_name, model_name, languages):
    features = pd.read_csv('repository/features/language_features.csv')
    scores = print_exp_results(benchmark_name, model_name, features, languages, return_scores=True)
    iso_scores = {transformed_to_iso[lan]: score for lan, score in scores.items()}

    results = features.merge(pd.DataFrame(list(iso_scores.items()), columns=['language', 'accuracy']), on='language', how='left')

    results = results[results['language'] != 'eng']

    print(results)
    plot_results(results)

run_experiment(languages_large_transformed_filtered)
evaluate_experiments(benchmark_name, model_name, languages_large_transformed_filtered)
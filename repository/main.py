from utils import read, get_subset
import random
import os
from classes import MultilingualBenchmark
from logger import logger

os.chdir('multilingual-llm-trustworthiness')

random.seed(42)
config = read('repository/config.json')
 
# SERIES OF SUBSETS (differing from debug to full to other sets)
language_subset_debug = [
    'zh-CN', 'en', 'hi', 'nl', 'fi'
]

languages_subset_small = [
    'fr', 'nl', 'es', 'cy', 'en',
    'sw', 'hu', 'bn', 'de', 'ru'
]

# EN English
# FR French
# DE German
# CS Czech
# IS Icelandic
# ZH Chinese
# JA Japanese
# RU Russian
# UK Ukranian
# HA Hausa

known_subset = [
    'en',     # English
    'fr',     # French
    'de',     # German
    'cs',     # Czech
    'is',     # Icelandic
    'zh-CN',  # Chinese
    'ja',     # Japanese
    'ru',     # Russian
    'uk',     # Ukrainian
    'ha'      # Hausa
]

# (ChrF, BLEU EN --> target, ChrF, BLEU target --> EN)
known_subset_metrics = {
    'de': (60.2, 31.8, 56.8, 30.9),
    'zh-CN': (34.6, 38.3, 56.0, 25.0),
    'ru': (54, 27.5, 64.6, 38.5),
    'fr': (58.9, 35.6, 65.7, 42.5),
    'cs': (),
    'is': (),
}

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
iso_to_transformed = {iso: trans for iso, trans in zip(language_subset_iso, language_subset_transformed)}
transformed_to_iso = {value: key for key, value in iso_to_transformed.items()}

benchmark = MultilingualBenchmark(model_name='gpt-4o-mini', run_name='debug', languages=language_subset_transformed, config=config)

# global debug parameter. Set to False if running main with an already existing .json results file.
RUN_EXPERIMENTS = False

if RUN_EXPERIMENTS:
    logger.info("Now running experiments.")
    benchmark_path = config.get('benchmark', {}).get('path', {})
    subset = get_subset(benchmark_path, n="Inf")
    benchmark.load_benchmark(subset)
    benchmark.run(print_results=True, plot_results=False)
    benchmark.to_json(f'repository/benchmarks/results/{benchmark.run_name}_{benchmark.model_name}.json')
else:
    benchmark = benchmark.from_json(r'repository\benchmarks\results\debug_gpt-4o-mini.json', 
                                    model_name='gpt-4o-mini', 
                                    run_name='debug', 
                                    languages=language_subset_transformed, 
                                    config=config)

    benchmark.print_results()
    benchmark.plot_results(iso_to_transformed, transformed_to_iso)
    
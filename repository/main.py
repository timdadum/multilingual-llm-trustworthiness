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

# ISO 639-3 codes for languages (three-letter codes)
language_subset_iso = [
    'eng',  # English
    'rus',  # Russian
    'deu',  # German
    'jpn',  # Japanese
    'spa',  # Spanish
    'fra',  # French
    'zho',  # Chinese (Mandarin)
    'ita',  # Italian
    'por',  # Portuguese
    'nld',  # Dutch
    'vie',  # Vietnamese
    'ind',  # Indonesian
    'arb',  # Arabic
    'swe',  # Swedish
    'hun',  # Hungarian
    'fin',  # Finnish
    'hin',  # Hindi
    'ben',  # Bengali
    'lav',  # Latvian
    'urd',  # Urdu
    'cym',  # Welsh
    'swh',  # Swahili
    'amh',  # Amharic
    'zul',  # Zulu
    'mri',  # Maori
]

# ISO 639-1 codes for languages (two-letter codes, with some variations like zh-CN for Chinese)
language_subset_transformed = [
    'en',     # English
    'ru',     # Russian
    'de',     # German
    'ja',     # Japanese
    'es',     # Spanish
    'fr',     # French
    'zh-CN',  # Chinese (Mandarin)
    'it',     # Italian
    'pt',     # Portuguese
    'nl',     # Dutch
    'vi',     # Vietnamese
    'id',     # Indonesian
    'ar',     # Arabic
    'sv',     # Swedish
    'hu',     # Hungarian
    'fi',     # Finnish
    'hi',     # Hindi
    'bn',     # Bengali
    'lv',     # Latvian
    'ur',     # Urdu
    'cy',     # Welsh
    'sw',     # Swahili
    'am',     # Amharic
    'zu',     # Zulu
    'mi',     # Maori
]

# Create convenient mappings
iso_to_transformed = {iso: trans for iso, trans in zip(language_subset_iso, language_subset_transformed)}
transformed_to_iso = {value: key for key, value in iso_to_transformed.items()}

benchmark = MultilingualBenchmark(benchmark_name='truthfulqa', 
                                  model_name='gpt-4o-mini', 
                                  run_name='gpt-4o-mini', 
                                  languages=language_subset_transformed, 
                                  config=config)

# global debug parameter. Set to False if running main with an already existing .json results file.
RUN_EXPERIMENTS = True

if RUN_EXPERIMENTS:
    logger.info("Now running experiments.")
    benchmark_path = config.get('benchmark', {}).get('path', {})
    subset = get_subset(benchmark_path, n=64)
    benchmark.load_benchmark(subset)
    benchmark.run(print_results=True, plot_results=True)
    benchmark.write_to_json(f'repository/benchmarks/results/{benchmark.benchmark_name}_{benchmark.model_name}.json')
else:
    benchmark = benchmark.from_json(r'repository\benchmarks\results\truthfulqa_gpt-4o-mini.json',
                                    benchmark_name='truthfulqa', 
                                    model_name='gpt-4o-mini', 
                                    run_name='gpt-4o-mini', 
                                    languages=language_subset_transformed, 
                                    config=config)

    benchmark._get_metrics()
    benchmark.print_results()
    benchmark.plot_results()

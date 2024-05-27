from deep_translator import GoogleTranslator
from utils import write
from tqdm import tqdm
import re

def _translate_sample_to_en(sample, translator):
    # Catch incorrect target language
    if translator.target != 'en':
        raise ValueError("Translator target is not 'en'. Please change.")
    
    # Translate question, answer and output back to English
    sample['Question']

    

def translate_benchmark(translator, benchmark, save=True, benchmark_name=None):
    """Assumes list of dictionaries format, e.g.:
    [
        {
            "Question": "Who was the man behind The Chipmunks?"
            "Answer": "David Seville"
        },
        {
            "Quesiton": ...
            "Answer": 
        }
        etc.
    ]
    
    and translates using translator (which is already set to target language)"""
    translated_benchmark = [{'target_language': translator.target}]
    for i, sample in enumerate(tqdm(benchmark, desc="Translating... This may take a while...")):
        try:
            q = benchmark[i]["Question"]
            a = benchmark[i]["Answer"]
            query = f'{q} @ {a}'
            
            # Get the translation, separate by separator token
            translation = translator.translate(query)
            split_translation = re.split(r'@', translation)
            qt, at = split_translation[0], split_translation[1]

            translated_benchmark.append(
                {
                "Question": qt,
                "Answer": at
                }
            )
        except Exception as e:
            print("An error occurred during translating. Ignoring datapoint...")
    print("Translation succesful")
    
    if save:
        filename = f'{benchmark_name}_{translator.target}'
        try: 
            print(f"Writing to {filename}")
            path = f'repository/benchmarks/translated/{filename}'
            write(path, translated_benchmark)
        except:
            raise ValueError("Please set benchmark name")
        
    return translated_benchmark


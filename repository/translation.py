from deep_translator import GoogleTranslator
from utils import write
from tqdm import tqdm

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
        q = benchmark[i]["Question"]
        a = benchmark[i]["Answer"]
        translated_benchmark.append(
            {
            "Question": translator.translate(q),
            "Answer": translator.translate(a)
            }
        )
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


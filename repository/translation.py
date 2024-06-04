from deep_translator import GoogleTranslator
from utils import write, _threadpool, filter_data_by_language
from tqdm import tqdm
import re

PLACEHOLDER = "<UN1QU3_D3L1M1T3R_PL4C3H0LD3R>"

def translate_data(data, source_language, target_languages, mode='qa', batch_size=64, save=True, benchmark_name=None):
    """Assumes list of dictionaries format, e.g.:
    [
        {
            "Question": "Who was the man behind The Chipmunks?"
            "Answer": "David Seville"
        },
        {
            "Question": ...
            "Answer": 
        }
        etc.
    ]
    
    and adds translations (e.g. "Answer_it" for Italian answer) using target-set translator
    
    NOTE: IS ONE-TO-MANY. 
    """
    # Catch single string input case
    if isinstance(target_languages, str):
        target_languages = [target_languages]

    for target_language in target_languages:
        translator = GoogleTranslator(target=target_language)
        for i in tqdm(range(0, len(data), batch_size), desc=f"Translating to {target_language}... This may take a while..."):
            batch = data[i:i + batch_size]
            
            # Filter out invalid samples
            valid_samples = []
            original_indices = []
            
            for index, sample in enumerate(batch):
                try:
                    _prepare_query(sample, source_language, mode)  # Just to check if the sample is valid
                    valid_samples.append(sample)
                    original_indices.append(index)
                except KeyError as e:
                    print(f'Missing data in sample, skipping sample... {e}')
            
            # Prepare queries for valid samples
            queries = [_prepare_query(sample, source_language, mode) for sample in valid_samples]
            
            # Translate the valid queries
            try:
                translated_queries = translator.translate_batch([q.replace('<<<|||>>>', PLACEHOLDER) for q in queries])
                translations = [t.replace(PLACEHOLDER, ' <<<|||>>> ') for t in translated_queries]
                
                # Reassign translations to the original batch
                for idx, translation in zip(original_indices, translations):
                    try:
                        translated_sample = _process_translation(batch[idx], translation, target_language, mode)
                        data[i + idx] = translated_sample
                    except Exception as e:
                        print(f'Error processing translation for sample {i + idx}, skipping sample... {e}')
            
            except Exception as e:
                print(f'Error translating batch, skipping batch... {e}')
                continue
    
    print("Translation successful")
    
    if save:
        if not benchmark_name:
            raise ValueError("Please set benchmark name")
        filename = f'{benchmark_name}_translated'
        try:
            print(f"Writing to {filename}")
            path = f'repository/benchmarks/translated/{filename}'
            write(path, data)
        except Exception as e:
            raise ValueError(f"Error writing file: {e}")
        
    return data

def _prepare_query(sample, source_language, mode):
    parts = []
    if 'q' in mode:
        q = sample[f"Question_{source_language}"]
        if not q.endswith('?'):
            q += '?'
        parts.append(q)
    if 'a' in mode:
        parts.append(sample[f"Answer_{source_language}"])
    if 'o' in mode and f"Output_{source_language}" in sample:
        parts.append(sample[f"Output_{source_language}"])
    if 'r' in mode and f"Result_{source_language}" in sample:
        parts.append(sample[f"Result_{source_language}"])

    return ' <<<|||>>> '.join(parts)

def _process_translation(sample, translation, target_language, mode):
    split_translation = re.split(r'\s*<<<\|+\|\|+\s*>>>\s*', translation)
    keys = []
    if 'q' in mode:
        keys.append(f"Question_{target_language}")
    if 'a' in mode:
        keys.append(f"Answer_{target_language}")
    if 'o' in mode and "Output" in sample:
        keys.append(f"Output_{target_language}")
    if 'r' in mode and "Result" in sample:
        keys.append(f"Result_{target_language}")

    if len(split_translation) != len(keys):
        raise ValueError("Translation split error. Resulting translation: ", split_translation)

    for key, translated_part in zip(keys, split_translation):
        sample[key] = translated_part.strip()

    return sample

DEBUG = False

if DEBUG:
    dummy_data = [
        {
            "Question": "How old is the earth?",
            "Answer": "Really old dude",
        },
        {
            "Question": "How tall is the eiffel tower?",
            "Answer": "Really tall dude"
        }
    ]
    dummy_langs = ['ar', 'nl', 'fr']

    output = translate_data(dummy_data, dummy_langs, save=False)
    print(output)
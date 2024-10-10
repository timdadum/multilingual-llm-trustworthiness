from logger import logger
import time
import json
import unicodedata

def prepare_translation_batches(batches, data_type):
    """Prepares batched benchmark for translation"""
    def _prepare_single_translation_batch(batch, data_type):
        if data_type == "question":
            return [sample._to_question_str() for sample in batch]
        else:
            # data_type == "output":
            return [sample._to_output_str() for sample in batch] 

    return [_prepare_single_translation_batch(batch, data_type) for batch in batches]

def translate_batch(index, batch, results, delay, translator):
    """
    Function that performs translation for a batch of strings
    index: The position of the sample in the original list.
    batch: List of strings to translate
    results: The list where translated texts will be stored.
    delay: Delay in seconds before sending the request.
    translator: Translator object that will handle the translation.
    data_type: The type of data being translated ('output' or 'question').
    """
    try:
        # Apply delay before making the request
        if delay > 0:
            time.sleep(delay)
        
        translated_batch = translator.translate_batch(batch)
        results[index] = translated_batch

    except Exception as e:
        results[index] = None  # Handle any translation failure
        logger.warning(f"Translation failed for a batch: {e}")
        return -1

def load_translation_table(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def character_translate(text, translation_table):
    # Normalize text for consistency
    normalized_text = unicodedata.normalize('NFKC', text)
    
    # Translate character-by-character
    translated_text = []
    for char in normalized_text:
        # If the character is in the mapping, translate it; otherwise, use �
        translated_text.append(translation_table.get(char, '�'))
    
    return ''.join(translated_text)
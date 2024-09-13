import re
from logger import logger
import httpx
import asyncio

def _format_translation(samples, target, mode='q'):
    """
    Formats a list of samples for translation.
    
    Args:
        samples (list): List of samples to be translated
        target (str): Target language conforming to deep_translator.GoogleTranslator docs, e.g., 'cy' for Welsh, 'hi' for Hindi
        mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language (backtranslation)
    Returns:
        A single query string ready to be translated
    """
    return [_prepare_query(sample, target, mode) for sample in samples]

async def translate_language_async(translator, target, samples, delay=5, mode='q'):
    """
    Asynchronously translates all samples for a given language with a delay between languages.
    
    Args:
        translator (deep_translator.GoogleTranslator): Translator object
        target (str): Target language conforming to deep_translator.GoogleTranslator docs, e.g., 'cy' for Welsh, 'hi' for Hindi
        samples (list): List of samples to be translated
        delay (int): Time in seconds to wait between processing each language
        mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language (back-translation)
    Returns:
        A list of translated texts
    """
    formatted_samples = _format_translation(samples, target, mode)
    
    async with httpx.AsyncClient() as client:
        try:
            # Sending the translation request for all samples
            translated_samples = await asyncio.to_thread(translator.translate_batch, formatted_samples)
            print(f"Translated {len(formatted_samples)} samples for language {target}.")
            await asyncio.sleep(delay)  # Introduce delay between each language
            return translated_samples
        except Exception as e:
            logger.error(f"Exception found in translating samples for language {target}: {e}. Returning None.")
            return None

def _prepare_query(sample, target, mode='q', delimiter='|'):
    """
    Preprocesses a sample for translation.
    
    Args:
        sample (Sample): Sample object
        target (str): Target language conforming to deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
        mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language (backtranslation)
    Returns:
        A sample represented as string, ready to be translated
    """

    # Assemble query as parts (question, answer, output) which are concatenated by a separator
    parts = []
    try:
        if mode == 'q':
            q = sample._to_question_str(language='en')
            parts.append(q)
        elif mode == 'qo':
            parts.append(sample._to_question_str(language=target))
            parts.append(sample._to_output_str(language=target))
            
        return delimiter.join(parts)
    except Exception as e:
        logger.error(f"Error preparing query: {e}")

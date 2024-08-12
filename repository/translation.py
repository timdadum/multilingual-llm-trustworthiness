import re
from logger import logger

def _format_translation_batch(batch, target, mode='q'):
    """
    Formats a translation batch.
    
    Args:
        batch (list, str): Contains list of strings to be translated
        target (str): Target language conform deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
        mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language (backtranslation)
    Returns:
        A transformed list of queries ready to be translated (Google Translate) 
    """
    return [_prepare_query(sample, target, mode) for sample in batch]
  
def translate_batch(translator, target, batch, mode='q'):
    """
    Translates a formatted batch.
    
    Args:
        translator (deep_translator.GoogleTranslator): translator object
        target (str): Target language conform deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
        mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language (backtranslation)
        batch (list, str): Contains list of formatted queries to be translated
    Returns:
        A translated list of queries 
    """
    translation_batch = _format_translation_batch(batch, target, mode)
    try:
        return translator.translate_batch(translation_batch)
    except Exception as e:
        logger.error(f"Exception found in translating batch: {e}. Returning None.")
        return None

def _prepare_query(sample, target, mode='q', delimiter='|'):
    """
    Preprocesses a sample for translation.
    
    Args:
        sample (Sample): Sample object
        target (str): Target language conform deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
        mode (str): 
            - 'q' for translation of questions, 
            - 'qo' for translation of questions to target and the output given in the target language (backtranslation)
        batch (list, str): Contains list of formatted queries to be translated
    Returns:
        A sample represented as string, ready to be translated
    """

    # Assemble query as parts (question, answer, output) which are concatenated by a separator (<S3P>)
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
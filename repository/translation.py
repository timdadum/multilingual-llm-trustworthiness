from logger import logger
import time

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
        logger.warning(f"Translation failed for a batch")
        return -1


# PROBABLY REPLACED BY GENERAL _THREADPOOL() CALL
# async def translate_async(translator, text_batches):
#     """
#     translator = translator object (deeptranslator.GoogleTranslator)
#     text_batches = list of text batches (each batch is a list of strings) to be translated in parallel
#     """
#     # Function to translate a single batch
#     async def translate_single_batch(batch):
#         return await asyncio.to_thread(translator.translate_batch, batch)

#     # Create tasks to translate each batch in parallel
#     tasks = [translate_single_batch(batch) for batch in text_batches]

#     # Run all batch translations concurrently and gather results
#     translated_batches = await asyncio.gather(*tasks)

#     # Flatten the list of lists into a single list of translated texts
#     # translated_texts = [text for batch in translated_batches for text in batch]

#     return translated_batches

# async def translate_single_batch(translator, text_batch):
#     """
#     translator = translator object (which contains a method called translate_batch, it's deeptranslator.GoogleTranslator)
#     text_batch = list of strings
#     """
#     translated_texts = await translator.translate_batch(text_batch)
#     return translated_texts

# async def translate_language_async(translator, target, samples, delay=5, mode='q'):
#     """
#     Asynchronously translates all samples for a given language with a delay between languages.
    
#     Args:
#         translator (deep_translator.GoogleTranslator): Translator object
#         target (str): Target language conforming to deep_translator.GoogleTranslator docs, e.g., 'cy' for Welsh, 'hi' for Hindi
#         samples (list): List of samples to be translated
#         delay (int): Time in seconds to wait between processing each language
#         mode (str): 
#             - 'q' for translation of questions, 
#             - 'qo' for translation of questions to target and the output given in the target language (back-translation)
#     Returns:
#         A list of translated texts
#     """
#     formatted_samples = _format_translation(samples, target, mode)
    
#     async with httpx.AsyncClient() as client:
#         try:
#             # Sending the translation request for all samples
#             translated_samples = await asyncio.to_thread(translator.translate_batch, formatted_samples)
#             print(f"Translated {len(formatted_samples)} samples for language {target}.")
#             await asyncio.sleep(delay)  # Introduce delay between each language
#             return translated_samples
#         except Exception as e:
#             logger.error(f"Exception found in translating samples for language {target}: {e}. Returning None.")
#             return None

# def _prepare_query(sample, target, mode='q', delimiter='|'):
#     """
#     Preprocesses a sample for translation.
    
#     Args:
#         sample (Sample): Sample object
#         target (str): Target language conforming to deep_translator.GoogleTranslator docs, e.g. 'cy' for Welsh, 'hi' for Hindi
#         mode (str): 
#             - 'q' for translation of questions, 
#             - 'qo' for translation of questions to target and the output given in the target language (backtranslation)
#     Returns:
#         A sample represented as string, ready to be translated
#     """

#     # Assemble query as parts (question, answer, output) which are concatenated by a separator
#     parts = []
#     try:
#         if mode == 'q':
#             q = sample._to_question_str(language='en')
#             parts.append(q)
#         elif mode == 'qo':
#             parts.append(sample._to_question_str(language=target))
#             parts.append(sample._to_output_str(language=target))
            
#         return delimiter.join(parts)
#     except Exception as e:
#         logger.error(f"Error preparing query: {e}")

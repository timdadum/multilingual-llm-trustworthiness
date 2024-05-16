from deep_translator import GoogleTranslator

translated = GoogleTranslator(source='auto', target='nl').translate("")

print(translated)
from deep_translator import GoogleTranslator
import json
from model import load_bloomz
from utils import read, write

with open('repository/config.json', 'r') as file:
    config = json.load(file)

translator = GoogleTranslator(source='auto', target=config.get('target', {}).get('language', {}))
tokenizer, model = load_bloomz()
data = read(config.get('benchmark', {}).get('path', {}))

print(data)

print(translator.translate("De rapen zijn gaar!!!"))
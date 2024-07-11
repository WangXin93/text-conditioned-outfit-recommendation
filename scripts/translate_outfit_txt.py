from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from urllib.parse import unquote
import string
import translators as ts
import translators.server as tss
import pickle
import os
import re
# from langdetect import detect, DetectorFactory, lang_detect_exception
# DetectorFactory.seed = 0

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def is_english(s):
    return s.isascii()

j = json.load(open('data/UIUC-polyvore/polyvore_outfits/polyvore_outfit_titles.json'))

save_path = 'en_outfit_urlName.json'
if os.path.exists(save_path):
    en_text = json.load(open(save_path))
else:
    en_text = {}

try:
    for k, v in tqdm(j.items()):
        if k in en_text:
            continue
        content = v['url_name']
        if not is_english(content):
            if content == 'Net-A-Porter.com' or content == 'MATCHESFASHION.COM':
                pass
            else:
                print(content)
                content = tss.google(content, to_language='en')
        en_text[k] = content
except Exception as e:
    print(e)
finally:
    print("data saved")
    json.dump(en_text, open(save_path, 'w'))

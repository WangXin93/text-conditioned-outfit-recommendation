from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from urllib.parse import unquote
import string
import pickle
import os
import re

def get_text(v):
    title = v['title']
    description = v['description']
    if len(title) > 0:
        content = title + '. ' + description
    else:
        content = description
    return content

model_name = 'distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(model_name)
model = model.cuda()

j = json.load(open('data/UIUC-polyvore/polyvore_outfits/polyvore_item_metadata.json'))

save_path = f'./encoded_itemTitleDescription_{model_name}.pkl'
if os.path.exists(save_path):
    encoded = pickle.load(open(save_path, 'rb'))
else:
    encoded = {}

try:
    for k, v in tqdm(j.items()):
        if k in encoded:
            continue
        url_name = v['url_name']
        url_name = unquote(url_name)
        content = get_text(v)
        out = model.encode(content)
        encoded[k] = out
except Exception as e:
    print(e)
finally:
    print("data saved")
    pickle.dump(encoded, open(save_path, 'wb'))

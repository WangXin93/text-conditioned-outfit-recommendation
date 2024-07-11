from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from urllib.parse import unquote
import string
import pickle
import os
import re

model_name = 'distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(model_name)

model = model.cuda()

j = json.load(open('data/UIUC-polyvore/polyvore_outfits/polyvore_item_metadata.json'))

save_path = f'./encoded_category_{model_name}.pkl'
if os.path.exists(save_path):
    encoded = pickle.load(open(save_path, 'rb'))
else:
    encoded = {}

try:
    for k, v in tqdm(j.items()):
        semantic_category = v['semantic_category']
        content = semantic_category
        if content in encoded:
            continue
        out = model.encode(content)
        encoded[content] = out
except Exception as e:
    print(e)
finally:
    print("data saved")
    pickle.dump(encoded, open(save_path, 'wb'))
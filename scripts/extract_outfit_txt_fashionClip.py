from transformers import AutoProcessor, AutoModel
from pathlib import Path
import pickle
import torch
import json
import os
from tqdm import tqdm

j_outfit = json.load(open('en_outfit_title.json'))
j_outfitUrl = json.load(open('en_outfit_urlName.json'))

processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModel.from_pretrained("patrickjohncyh/fashion-clip")

model = model.to('cuda:0')

save_path = 'encoded_outfitUrlTitle_en_fashionClip.pkl'
if os.path.exists(save_path):
    text_feats = pickle.load(open(save_path), 'rb')
else:
    text_feats = {}

for i, (k, v) in tqdm(enumerate(j_outfit.items()), total=len(list(j_outfit.items()))):
    content = ' '.join([v, j_outfitUrl[k]])
    try:
        inputs = processor(text=content, return_tensors="pt", padding=True)
        inputs = inputs.to('cuda:0')
        inputs['input_ids'] = inputs['input_ids'][:, :77]
        inputs['attention_mask'] = inputs['attention_mask'][:, :77]
    except Exception as e:
        print(f'{k}: {content}')
        continue
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs)[0]
        text_feat = text_feat.detach().cpu().numpy() 
    text_feats[k] = text_feat
    
pickle.dump(text_feats, open(str(save_path), 'wb'))
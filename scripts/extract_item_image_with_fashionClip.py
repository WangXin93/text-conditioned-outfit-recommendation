from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import pickle
from tqdm import tqdm
from pathlib import Path
import json

j_item = json.load(open('data/UIUC-polyvore/polyvore_outfits/polyvore_item_metadata.json'))

processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")

model = AutoModel.from_pretrained("patrickjohncyh/fashion-clip")

model = model.to('cuda:0')

img_feats = {}
for i, (k, v) in tqdm(enumerate(j_item.items()), total=len(list(j_item.items()))):
    img_path = Path(f'data/UIUC-polyvore/polyvore_outfits/images') / (k+'.jpg')
    img = Image.open(str(img_path)).convert('RGB')
    inputs = processor(images=img, return_tensors="pt", padding=True)
    inputs = inputs.to('cuda:0')
    with torch.no_grad():
        img_feat = model.get_image_features(**inputs)[0]
        img_feat = img_feat.detach().cpu().numpy() 
    img_feats[k] = img_feat
pickle.dump(img_feats, open('img_feats_fashionClip.pkl', 'wb'))
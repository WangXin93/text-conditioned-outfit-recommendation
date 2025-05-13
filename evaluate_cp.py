import torch
from dataset import UIUCPolyvoreRetrievalDataset, UIUCPolyvorePredictionDataset
from outfit_transformer import (
    OutfitTransformerRetrieval, 
    LinearImageEncoder,
    LinearTextEncoder,
    MlpImageEncoder,
    MlpTextEncoder,
    OutfitTransformerPrediction
)
import argparse
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import time
import os
import random
import math
from torchvision.ops.focal_loss import sigmoid_focal_loss
from pathlib import Path
import torch.nn.functional as F
from contextlib import nullcontext
import inspect
from copy import deepcopy

# Change to other config for different settings
from config.cp_cond_hardneg import *
import config.cp_cond_hardneg as cfg


# print config parameters
for x in dir(cfg):
    if x.startswith('__'):
        continue
    if inspect.ismodule(cfg.__dict__.get(x)):
        continue
    if isinstance(cfg.__dict__.get(x), type):
        continue
    print('{:<40}{:<40}'.format(x, repr(cfg.__dict__.get(x))))

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# dataloader
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    torchvision.transforms.ToTensor(),
    normalize
])
train_dat = UIUCPolyvorePredictionDataset(
        datadir=datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='train',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
train_loader = DataLoader(train_dat, batch_size, shuffle=True, num_workers=num_workers)
test_dat = UIUCPolyvorePredictionDataset(
        datadir=datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='test',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
test_loader = DataLoader(test_dat, batch_size, shuffle=False, num_workers=num_workers)
print("Load data complete")

# model init
best_acc = -1
best_auc = -1
start_epoch = 0
model = OutfitTransformerPrediction(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size),
    nhead=nhead,
    num_layers=num_layers,
    use_outfit_txt=use_outfit_txt
)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
best_acc = checkpoint.get('best_acc', -1)
start_epoch = checkpoint['epoch']
print(f"Resuming training from {out_dir}, best_acc: {best_acc}, epoch: {start_epoch}")
model = model.to(device)

# compile the model
if compile_model:
    print("compiling the model ...")
    unoptimized_model = model
    model = torch.compile(model)

# evaluate model
model.eval()

# test CP auc
target_lst = []
output_lst = []
for outfit, target, set_id in tqdm(test_dat.compatibility_questions):
    outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)
    imgs, txts = test_dat.load_outfit(outfit)[:2]
    imgs, txts, mask = test_dat.pad_imgs_and_txts(imgs, txts)
    imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
    with torch.no_grad(), ctx:
        output = model(imgs.unsqueeze(0), txts.unsqueeze(0), mask.unsqueeze(0), outfit_txt=outfit_txt.unsqueeze(0))
    output_lst.append(output['logits'].item())
    target_lst.append(target)
output_lst = np.array(output_lst)
target_lst = np.array(target_lst)
auc = roc_auc_score(target_lst, output_lst)
print(f'AUC: {auc:.3f}')        

# test fitb       
n_questions = 0
correct = 0
for outfits, is_correct, set_id in tqdm(test_dat.fitb_questions, desc="compute FITB"):
    answer_score = []
    outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)
    for outfit in outfits:
        imgs, txts = test_dat.load_outfit(outfit)[:2]
        imgs, txts, mask = test_dat.pad_imgs_and_txts(imgs, txts)
        imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
        with torch.no_grad(), ctx:
            output = model(imgs.unsqueeze(0), txts.unsqueeze(0), mask.unsqueeze(0), outfit_txt=outfit_txt.unsqueeze(0))
        answer_score.append(output['logits'].item())
    answer_score = np.array(answer_score)
    correct += is_correct[np.argmax(answer_score)]
    n_questions += 1
acc = correct / n_questions
print(f'FITB ACC: {acc:.4f}')
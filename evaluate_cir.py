import torch
from dataset import UIUCPolyvoreRetrievalDataset
from outfit_transformer import (
    OutfitTransformerRetrieval, 
    LinearImageEncoder,
    LinearTextEncoder,
    MlpImageEncoder,
    MlpTextEncoder
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
import pickle
from contextlib import nullcontext
from copy import deepcopy
import inspect

# Change to other config for different settings
from config.cir_cond_hardneg import *
import config.cir_cond_hardneg as cfg

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
train_dat = UIUCPolyvoreRetrievalDataset(
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
train_loader = DataLoader(train_dat, batch_size=batch_size, shuffle=False)
test_dat = UIUCPolyvoreRetrievalDataset(
        datadir=datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='test',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False,
    )
test_loader = DataLoader(test_dat, batch_size=batch_size, shuffle=False)
print('data load complete')

model = OutfitTransformerRetrieval(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size) if use_outfit_txt else None,
    nhead=nhead,
    num_layers=num_layers,
    margin=margin,
    target_item_info='category',
    use_outfit_txt=use_outfit_txt
)
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
start_epoch = checkpoint['epoch']
best_acc = checkpoint.get('best_acc', -1)
model = model.to(device)
print(f"Resuming training from {out_dir}, best_acc: {best_acc:.4f}, epoch: {start_epoch}")

model.eval()

# evaluate compatibility prediction auc
# scores = []
# targets = []
# for outfit, target, set_id in tqdm(test_dat.compatibility_questions, desc='compatibility prediction'):
#     if len(outfit['items']) < 2:
#         continue

#     outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)
#     outfit_txt = outfit_txt.unsqueeze(0)

#     score_list = []
#     for i in range(1, len(outfit['items'])):
#         partial_imgs, partial_txts, _ = test_dat.load_outfit({'items': outfit['items'][:i]})
#         partial_imgs, partial_txts, mask = test_dat.pad_imgs_and_txts(partial_imgs, partial_txts)
#         partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
#         partial_imgs, partial_txts, mask = partial_imgs.unsqueeze(0), partial_txts.unsqueeze(0), mask.unsqueeze(0)

#         item_id = outfit['items'][i]['item_id']
#         target_item_category = torch.tensor(test_dat.category_feats[item_id]).to(device)
#         target_item_category = target_item_category.unsqueeze(0)
        
#         target_feat = test_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=False)
#         with torch.no_grad(), ctx:
#             out = model(
#                 partial_imgs, partial_txts, mask, 
#                 positive_category=target_item_category,
#                 outfit_txt=outfit_txt
#             )
#             score = torch.cosine_similarity(target_feat, out['logits'].detach())
#         score_list.append(score.item())
#     scores.append(np.mean(score_list))
#     targets.append(target)

# from sklearn.metrics import roc_auc_score

# print(f"AUC: {roc_auc_score(targets, scores)}")
# import sys; sys.exit()

# test fitb       
# n_questions = 0
# correct = 0
# for outfits, is_correct, set_id in tqdm(test_loader.dataset.fitb_questions):
#     partial_imgs, partial_txts, _ = test_loader.dataset.load_outfit({'items': outfits[0]['items'][:-1]})
#     partial_imgs, partial_txts, mask = test_loader.dataset.pad_imgs_and_txts(partial_imgs, partial_txts)

#     ans_imgs = []
#     ans_txts = []
#     ans_categories = []
#     for outfit in outfits:
#         ans_img, ans_txt, ans_category = test_loader.dataset.load_item(outfit['items'][-1])
#         ans_imgs.append(ans_img)
#         ans_txts.append(ans_txt)
#         ans_categories.append(ans_category)
#     ans_imgs = torch.stack(ans_imgs).to(device)
#     ans_txts = torch.stack(ans_txts).to(device)
#     ans_categories = torch.stack(ans_categories).to(device)

#     correct_ind = np.where(is_correct)[0][0]
#     positive_txt = ans_txts[correct_ind]
#     positive_category = ans_categories[correct_ind]

#     positive_txt = positive_txt.to(device)
#     positive_category = positive_category.to(device)
#     partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
            
#     outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)

#     with torch.no_grad():
#         with ctx:
#             out = model(
#                 partial_imgs.unsqueeze(0), partial_txts.unsqueeze(0), mask.unsqueeze(0), 
#                 positive_category=positive_category.unsqueeze(0),
#                 outfit_txt=outfit_txt.unsqueeze(0)
#             )
#             ans_imgs_emb = model.img_encoder(ans_imgs)
#             ans_txts_emb = model.text_encoder(ans_txts)
#             ans_emb = torch.cat([ans_imgs_emb, ans_txts_emb], dim=-1)
#             dist = F.cosine_similarity(out['logits'], ans_emb, dim=-1)
#             ans = torch.argmax(dist)

#     if is_correct[ans]:
#         correct += 1
#     n_questions += 1

# acc = correct / n_questions

# print(f'FITB ACC: {acc:.4f}')

# test recall@topk
# generate database for each fine-grained category
distractors_id = dict()
distractors_feats = dict()
distractors_idSet = dict()
for k, v in tqdm(test_dat.fg2ims.items()):
    if k not in distractors_id:
        distractors_id[k] = list()
        distractors_feats[k] = list()
        distractors_idSet[k] = set()
    for set_id, item_lst in v.items():
        for item_id in item_lst:
            if item_id not in distractors_idSet[k]:
                distractors_idSet[k].add(item_id)
                distractors_id[k].append(item_id)
                distractors_feats[k].append(
                    test_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
                )
            if len(distractors_id[k]) >= 3000:
                break
        if len(distractors_id[k]) >= 3000:
            break
for k, v in tqdm(train_dat.fg2ims.items()):
    if k not in distractors_id:
        break
    for set_id, item_lst in v.items():
        for item_id in item_lst:
            if item_id not in distractors_idSet[k]:
                distractors_idSet[k].add(item_id)
                distractors_id[k].append(item_id)
                distractors_feats[k].append(
                    test_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
                )
            if len(distractors_id[k]) >= 3000:
                break
        if len(distractors_id[k]) >= 3000:
            break
# filter out fine-grained categories with less than 3000 distractors
distractors_id = {k:v for k, v in distractors_id.items() if len(v) >= 3000}
distractors_feats = {k:v for k, v in distractors_feats.items() if len(v) >= 3000}
distractors_id = {k:np.array(v) for k, v in distractors_id.items()}
distractors_feats = {k:torch.tensor(np.concatenate(v)) for k, v in distractors_feats.items()}
print(f'[{len(distractors_feats)}/{len(test_dat.fg2ims)}] fine-grained categories are selected.')

def retrieval_top_k(
        logits: torch.tensor, 
        distractors_feats: torch.tensor,
        distractors_id: np.ndarray,
        k: int
    ):
    score = torch.cosine_similarity(logits, distractors_feats)
    values, indices = torch.sort(score, descending=True)
    topIdx = distractors_id[indices.numpy()]
    return topIdx[:k]

count = 0
k = [1, 3, 5, 10, 30, 50]
correct = [0] * 6
for outfits, is_correct, set_id in tqdm(test_dat.fitb_questions, desc='compute recall@topk'):
    target_item_id = outfits[np.where(is_correct)[0][0]]['items'][-1]['item_id']
    target_item_fg = test_dat.im2fg[target_item_id]
    if target_item_fg not in distractors_feats:
        continue

    partial_imgs, partial_txts, _ = test_dat.load_outfit({'items': outfits[0]['items'][:-1]})
    partial_imgs, partial_txts, mask = test_dat.pad_imgs_and_txts(partial_imgs, partial_txts)
    partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
    target_item_category = torch.tensor(test_dat.category_feats[target_item_id])
    target_item_category = target_item_category.to(device)
    outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)

    count += 1
    with torch.no_grad(), ctx:
        out = model(
            partial_imgs.unsqueeze(0), partial_txts.unsqueeze(0), mask.unsqueeze(0), 
            positive_category=target_item_category.unsqueeze(0),
            outfit_txt=outfit_txt.unsqueeze(0)
        )
        logits = out['logits'].detach().cpu().float()
        cur_dist_id = deepcopy(distractors_id[target_item_fg])
        cur_dist_feats = deepcopy(distractors_feats[target_item_fg])
        if target_item_id not in cur_dist_id:
            cur_dist_id = np.concatenate([cur_dist_id[:-1], np.array([target_item_id])])
            cur_dist_feats = torch.cat([cur_dist_feats[:-1, :], 
                                        test_dat.extract_emb({'item_id':target_item_id}, model, device, ctx, to_numpy=False).detach().cpu()], dim=0)
        topIdx = retrieval_top_k(logits, cur_dist_feats, cur_dist_id, k[-1])
        for i, k_ in enumerate(k):
            if target_item_id in topIdx[:k_]:
                correct[i] += 1

recall =  [c / count for c in correct]
for k_, recall_ in zip(k, recall):
    print(f'Recall@top{k_}: {recall_*100:.4f}%')

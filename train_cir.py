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
from contextlib import nullcontext
import inspect
from copy import deepcopy

from config.cir_cond_hardneg import *
import config.cir_cond_hardneg as cfg

from utils import set_logger
logger = set_logger(out_dir)

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
train_loader = DataLoader(train_dat, batch_size, shuffle=True, num_workers=num_workers)
valid_dat = UIUCPolyvoreRetrievalDataset(
        datadir=datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='valid',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
valid_loader = DataLoader(valid_dat, batch_size, shuffle=False, num_workers=num_workers)
logger.info("Load data complete")

# model init
best_acc = -1
best_auc = -1
start_epoch = 0
model = OutfitTransformerRetrieval(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size) if use_outfit_txt else None,
    nhead=nhead,
    num_layers=num_layers,
    margin=margin,  
    target_item_info='category',
    use_outfit_txt=use_outfit_txt,
)
if init_from == 'scratch':
    logger.info("Initializing a new model from scratch")
    if pretrained_cp_model is not None:
        model.from_cp_pretriained(pretrained_cp_model)
elif init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    best_acc = checkpoint.get('best_acc', -1)
    start_epoch = checkpoint['epoch']
    logger.info(f"Resuming training from {out_dir}, best_acc: {best_acc}, epoch: {start_epoch}")
model = model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
if init_from == 'resume':
    scaler.load_state_dict(checkpoint["scaler"])

# optimizer
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=learning_rate)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)

# compile the model
if compile_model:
    logger.info("compiling the model ...")
    unoptimized_model = model
    model = torch.compile(model)

t0 = time.time()
global_iter_num = 1
for epoch in range(start_epoch+1, epochs+1):
    # train one epoch
    model.train()
    if epoch >= hard_negatives_start_epoch:
        train_loader.dataset.sample_hard_negative = True
        logger.info('Sampling hard negatives')
    else:
        train_loader.dataset.sample_hard_negative = False
        logger.info('Not sampling hard negatives')
    for iter_num, (partial_imgs,
                   partial_txts,
                   mask,
                   positive_img,
                   positive_txt,
                   positive_category,
                   negative_imgs,
                   negative_txts,
                   hard_negative_imgs,
                   hard_negative_txts,
                   outfit_txt) in enumerate(train_loader, start=1):
                
        partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
        positive_img, positive_txt = positive_img.to(device), positive_txt.to(device)
        positive_category = positive_category.to(device)
        negative_imgs, negative_txts = negative_imgs.to(device), negative_txts.to(device)
        outfit_txt = outfit_txt.to(device)
        with ctx:
            if epoch >= hard_negatives_start_epoch:
                hard_negative_imgs, hard_negative_txts = hard_negative_imgs.to(device), hard_negative_txts.to(device)
                out = model.forward(
                    partial_imgs, partial_txts, mask, 
                    positive_img, positive_txt, positive_category, 
                    negative_imgs, negative_txts,
                    hard_negative_imgs, hard_negative_txts,
                    outfit_txt=outfit_txt
                )
            else:
                out = model.forward(
                    partial_imgs, partial_txts, mask, 
                    positive_img, positive_txt, positive_category, 
                    negative_imgs, negative_txts,
                    outfit_txt=outfit_txt
                )
        loss = out['loss']
                                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        global_iter_num += 1
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval_steps == 0 or iter_num == 1:
            logger.info(f"epoch [{epoch: 3}/{epochs}] "
                         f"iter {iter_num: 5} [{iter_num*batch_size: 6}/{len(train_dat)}]: "
                         f"lr {scheduler.get_last_lr()[0]:.4e} "
                         f"loss {loss.item():.4f}, "
                         f"time {dt*1000:.2f}ms")
            
    # decay learning rate if need
    if epoch % 10 == 0:
        scheduler.step()
    
    if epoch % eval_interval_epochs == 0:
        # evaluate model
        model.eval()

        # test fitb       
        n_questions = 0
        correct = 0

        for outfits, is_correct, set_id in tqdm(valid_loader.dataset.fitb_questions, desc="compute FITB"):
            partial_imgs, partial_txts, _ = valid_loader.dataset.load_outfit({'items': outfits[0]['items'][:-1]})
            partial_imgs, partial_txts, mask = valid_loader.dataset.pad_imgs_and_txts(partial_imgs, partial_txts)

            ans_imgs = []
            ans_txts = []
            ans_categories = []
            for outfit in outfits:
                ans_img, ans_txt, ans_category = valid_loader.dataset.load_item(outfit['items'][-1])
                ans_imgs.append(ans_img)
                ans_txts.append(ans_txt)
                ans_categories.append(ans_category)
            ans_imgs = torch.stack(ans_imgs).to(device)
            ans_txts = torch.stack(ans_txts).to(device)
            ans_categories = torch.stack(ans_categories).to(device)

            correct_ind = np.where(is_correct)[0][0]
            positive_txt = ans_txts[correct_ind]
            positive_category = ans_categories[correct_ind]

            positive_txt = positive_txt.to(device)
            positive_category = positive_category.to(device)
            partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)

            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id]).to(device)

            with torch.no_grad():
                with ctx:
                    out = model(
                        partial_imgs.unsqueeze(0), partial_txts.unsqueeze(0), mask.unsqueeze(0), 
                        positive_category=positive_category.unsqueeze(0),
                        outfit_txt=outfit_txt.unsqueeze(0)
                    )
                    ans_imgs_emb = model.img_encoder(ans_imgs)
                    ans_txts_emb = model.text_encoder(ans_txts)
                    ans_emb = torch.cat([ans_imgs_emb, ans_txts_emb], dim=-1)
                    dist = F.cosine_similarity(out['logits'], ans_emb, dim=-1)
                    ans = torch.argmax(dist)

            if is_correct[ans]:
                correct += 1
            n_questions += 1

        acc = correct / n_questions

        logger.info(f'FITB ACC: {acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch
            }
            logger.info(f"saving best checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt.pt'))

        if always_save_checkpoint:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch
            }
            logger.info(f"saving current epoch checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{epoch}.pt'))

        # evaluate recall@topk
        # generate database for each fine-grained category
        distractors_id = dict()
        distractors_feats = dict()
        distractors_idSet = dict()
        for k, v in tqdm(valid_dat.fg2ims.items()):
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
                            valid_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
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
                            valid_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
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
        logger.info(f'[{len(distractors_feats)}/{len(valid_dat.fg2ims)}] fine-grained categories are selected.')

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
        k = [10, 30, 50]
        correct = [0] * 3
        for outfits, is_correct, set_id in tqdm(valid_dat.fitb_questions, desc='compute recall@topk'):
            target_item_id = outfits[np.where(is_correct)[0][0]]['items'][-1]['item_id']
            target_item_fg = valid_dat.im2fg[target_item_id]
            if target_item_fg not in distractors_feats:
                continue

            partial_imgs, partial_txts, _ = valid_dat.load_outfit({'items': outfits[0]['items'][:-1]})
            partial_imgs, partial_txts, mask = valid_dat.pad_imgs_and_txts(partial_imgs, partial_txts)
            partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
            target_item_category = torch.tensor(valid_dat.category_feats[target_item_id])
            target_item_category = target_item_category.to(device)
            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id])
            outfit_txt = outfit_txt.to(device)

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
                                                valid_dat.extract_emb({'item_id':target_item_id}, model, device, ctx, to_numpy=False).detach().cpu()], dim=0)
                topIdx = retrieval_top_k(logits, cur_dist_feats, cur_dist_id, k[-1])
                for i, k_ in enumerate(k):
                    if target_item_id in topIdx[:k_]:
                        correct[i] += 1

        recall =  [c / count for c in correct]
        for k_, recall_ in zip(k, recall):
            logger.info(f'Recall@top{k_}: {recall_*100:.4f}%')
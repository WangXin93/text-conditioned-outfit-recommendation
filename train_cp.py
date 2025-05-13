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

from ipdb import iex, launch_ipdb_on_exception
os.environ['PYTHONBREAKPOINT'] = 'ipdb.set_trace'

from config.cp_cond_hardneg import *
import config.cp_cond_hardneg as cfg

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
    logger.info('{:<40}{:<40}'.format(x, repr(cfg.__dict__.get(x))))

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
valid_dat = UIUCPolyvorePredictionDataset(
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
model = OutfitTransformerPrediction(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size) if use_outfit_txt else None,
    nhead=nhead,
    num_layers=num_layers,
    use_outfit_txt=use_outfit_txt
)
if init_from == 'scratch':
    logger.info("Initializing a new model from scratch")
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
    for iter_num, (outfit_txt,
                   imgs, txts, mask,
                   negative_imgs, negative_txts, negative_mask,
                   hard_negative_imgs, hard_negative_txts, hard_negative_mask) in enumerate(train_loader, start=1):
                
        imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
        negative_imgs, negative_txts, negative_mask = negative_imgs.to(device), negative_txts.to(device), negative_mask.to(device)
        outfit_txt = outfit_txt.to(device)
        with ctx:
            if epoch >= hard_negatives_start_epoch:
                hard_negative_imgs, hard_negative_txts, hard_negative_mask = \
                    hard_negative_imgs.to(device), hard_negative_txts.to(device), hard_negative_mask.to(device)
                out_hard_neg = model.forward(
                    hard_negative_imgs, hard_negative_txts, hard_negative_mask, 
                    outfit_txt=outfit_txt,
                    target=torch.zeros((hard_negative_mask.shape[0], 1)).to(device)
                )
            out = model.forward(
                imgs, txts, mask, 
                outfit_txt=outfit_txt,
                target=torch.ones((imgs.shape[0], 1)).to(device)
            )
            out_neg = model.forward(
                negative_imgs, negative_txts, negative_mask, 
                outfit_txt=outfit_txt,
                target=torch.zeros((negative_imgs.shape[0], 1)).to(device)
            )
        loss = out['loss'] + out_neg['loss']
        if epoch >= hard_negatives_start_epoch:
            loss += out_hard_neg['loss']
                                
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

        # test CP auc
        target_lst = []
        output_lst = []
        for outfit, target, set_id in tqdm(valid_dat.compatibility_questions):
            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id]).to(device)
            imgs, txts = valid_dat.load_outfit(outfit)[:2]
            imgs, txts, mask = valid_dat.pad_imgs_and_txts(imgs, txts)
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
        for outfits, is_correct, set_id in tqdm(valid_dat.fitb_questions, desc="compute FITB"):
            answer_score = []
            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id]).to(device)
            for outfit in outfits:
                imgs, txts = valid_dat.load_outfit(outfit)[:2]
                imgs, txts, mask = valid_dat.pad_imgs_and_txts(imgs, txts)
                imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
                with torch.no_grad(), ctx:
                    output = model(imgs.unsqueeze(0), txts.unsqueeze(0), mask.unsqueeze(0), outfit_txt=outfit_txt.unsqueeze(0))
                answer_score.append(output['logits'].item())
            answer_score = np.array(answer_score)
            correct += is_correct[np.argmax(answer_score)]
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
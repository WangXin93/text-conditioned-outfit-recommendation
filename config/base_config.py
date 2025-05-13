from pathlib import Path
import multiprocessing
import torch
import random
import numpy as np
from contextlib import nullcontext

# system
device_type = 'cuda'
device = torch.device('cuda:0') if 'cuda' in device_type else torch.device('cpu')
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile_model = False # use PyTorch 2.0 to compile the model to be faster
torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
random.seed(1338)
np.random.seed(1339)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# I/O config
# out_dir = Path('runs/outfit_transformer_encoder_title_description_outfitUrlTitle')
# out_dir = Path('runs/outfit_transformer_prediction_title_description_outfitUrlTitle_hardneg')
# out_dir = Path('runs/outfit_transformer_prediction_title_description_outfitUrlTitle_hardneg')
# if not out_dir.exists():
#     out_dir.mkdir(parents=True, exist_ok=True)
log_interval_steps = 10
eval_interval_epochs = 1 # interval epoch between evaluation
init_from = 'scratch' # or 'resume' or 'scratch'
# cp_pretrained = 'runs/outfit_transformer_url_description'
always_save_checkpoint = True # if True, always save a checkpoint after evaluation

# model config
img_inp_size=512
img_emb_size=64
txt_inp_size=512
txt_emb_size=64
outfit_txt_inp_size=512
nhead=16
num_layers=3
dropout=0.1
use_outfit_txt=True

# data config
polyvore_split = 'disjoint'
max_item_len= 16 if polyvore_split=='disjoint' else 19 # 16 for disjoint, 19 for nondisjont
img_size = 224
num_workers = multiprocessing.cpu_count()
batch_size = 50 if 'float16' in dtype else 30
# datadir = '/mnt/22600F38600F1269/public_dataset/UIUC-polyvore/'
datadir = 'E:\\Datasets\\UIUC-polyvore'
txt_feat_path = './encoded_title_description_distiluse-base-multilingual-cased-v2.pkl'
img_feat_path = './img_feats_fashionClip.pkl'
category_feat_path = './encoded_category_distiluse-base-multilingual-cased-v2.pkl'
outfit_feat_path = './encoded_outfitUrlTitle_en_fashionClip.pkl'

# optimizer config
epochs = 100
learning_rate = 5e-5
# gradient_accumulation_steps = 5 # used to simulate larger batch sizes
margin = 0.3

# learning rate decay settings
decay_lr = True
# lr_decay_iters = (16995 // batch_size * 60) if polyvore_split=='disjoint' else (53306 // batch_size * 60)
# min_lr = 1e-6
# warmup_iters = 200
hard_negatives_start_epoch = 40

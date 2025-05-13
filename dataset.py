import os
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

from ipdb import iex, launch_ipdb_on_exception
os.environ['PYTHONBREAKPOINT'] = 'ipdb.set_trace'


def parse_iminfo(question, im2index, id2im, gt = None):
    """ Maps the questions from the FITB and compatibility tasks back to
        their index in the precomputed matrix of features

        question: List of images to measure compatibility between
        im2index: Dictionary mapping an image name to its location in a
                  precomputed matrix of features
        gt: optional, the ground truth outfit set this item belongs to
    """
    questions = []
    is_correct = np.zeros(len(question), np.bool_)
    for index, im_id in enumerate(question):
        set_id = im_id.split('_')[0]
        if gt is None:
            gt = set_id

        im = id2im[im_id]
        questions.append((im2index[im], im))
        is_correct[index] = set_id == gt

    return questions, is_correct, gt


def load_compatibility_questions(fn, im2index, id2im):
    """ Returns the list of compatibility questions for the
        split """
    with open(fn, 'r') as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        compat_question, _, gt = parse_iminfo(data[1:], im2index, id2im)
        outfit = {'items':[]}
        for index, item in enumerate(compat_question, start=1):
            outfit['items'].append({'item_id': item[1], 'index': index})
        compatibility_questions.append((outfit, int(data[0]), gt))

    return compatibility_questions


def load_fitb_questions(fn, im2index, id2im):
    """ Returns the list of fill in the blank questions for the
        split """
    data = json.load(open(fn, 'r'))
    questions = []
    for item in data:
        question = item['question']
        q_index, _, gt = parse_iminfo(question, im2index, id2im) # gt is set_id
        answer = item['answers']
        a_index, is_correct, _ = parse_iminfo(answer, im2index, id2im, gt)

        # convert question and answer to outfit
        outfits = []
        for item_ans in a_index:
            outfit = {'items': []}
            for item in q_index:
                outfit['items'].append({'item_id': item[1]})
            outfit['items'].append({'item_id': item_ans[1]})
            outfits.append(outfit)
        questions.append((outfits, is_correct, gt))
        
    return questions


class UIUCPolyvorePredictionDataset(Dataset):
    """Dataset for compatiblity prediction and FITB. It generates negative outfits 
    and positive outfits at the same time during training. It can also load 
    predefined compatiblity questions and FITB questions
    """
    def __init__(self,
            datadir=None,
            split='train',
            polyvore_split='disjoint',
            img_feat_path=None,
            txt_feat_path=None,
            category_feat_path=None,
            outfit_feat_path=None,
            max_item_len=16,
            transform=None,
            num_negative=10,
            sample_hard_negative=False
        ):
        rootdir = os.path.join(datadir, 'polyvore_outfits', polyvore_split)
        meta_data_path = os.path.join(datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
        meta_data = json.load(open(meta_data_path, 'r'))
        self.impath = os.path.join(datadir, 'polyvore_outfits', 'images')
        self.is_train = split == 'train'
        data_json = os.path.join(rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r'))
        txt_feats = pickle.load(open(txt_feat_path, 'rb'))
        if img_feat_path is not None:
            img_feats = pickle.load(open(img_feat_path, 'rb'))
        else:
            img_feats = None
        if category_feat_path is not None:
            category_feats = pickle.load(open(category_feat_path, 'rb'))
        else:
            category_feats = None
        if outfit_feat_path is not None:
            outfit_feats = pickle.load(open(outfit_feat_path, 'rb'))
        else:
            outfit_feats = None

        # get list of images and make a mapping used to quickly organize the data
        im2type = {}
        category2ims = {}
        imnames = set()
        id2im = {}
        fg2ims = {}
        im2fg = {}
        im2set = defaultdict(set)
        set2data = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            set2data[outfit_id] = outfit
            for item in outfit['items']:
                im = item['item_id']
                category = meta_data[im]['semantic_category']
                fg = meta_data[im]['category_id']
                im2type[im] = category
                im2fg[im] = fg
                im2set[im].add(outfit_id)

                if category not in category2ims:
                    category2ims[category] = {}
                if fg not in fg2ims:
                    fg2ims[fg] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []
                if outfit_id not in fg2ims[fg]:
                    fg2ims[fg][outfit_id] = []

                category2ims[category][outfit_id].append(im)
                fg2ims[fg][outfit_id].append(im)
                id2im['%s_%i' % (outfit_id, item['index'])] = im
                imnames.add(im)
        imnames = list(imnames)
        im2index = {}
        for index, im in enumerate(imnames):
            im2index[im] = index

        self.rootdir = rootdir
        self.split = split
        self.data = outfit_data
        self.imnames = imnames
        self.im2type = im2type
        self.im2set = im2set
        self.set2data = set2data
        self.transform = transform
        self.category2ims = category2ims
        self.id2im = id2im
        self.fg2ims = fg2ims
        self.im2fg = im2fg
        self.txt_feats = txt_feats
        self.img_feats = img_feats
        self.category_feats = category_feats
        self.outfit_feats = outfit_feats
        self.max_item_len = max_item_len
        self.num_negative = num_negative
        self.sample_hard_negative = sample_hard_negative

        self.compatibility_questions = load_compatibility_questions(
            os.path.join(rootdir, f'compatibility_{split}.txt'),
                im2index,
                id2im)
        self.fitb_questions = load_fitb_questions(
            os.path.join(rootdir, f'fill_in_blank_{split}.json'),
            im2index,
            id2im)

    def load_outfit(self, outfit):
        imgs = []
        txts = []
        categories = []
        for item in outfit['items']:
            img, txt, category = self.load_item(item)
            imgs.append(img)
            txts.append(txt)
            categories.append(category)
        if self.transform is not None:
            if len(imgs) > 0:
                imgs = torch.stack(imgs)
        if len(txts) > 0:
            txts = torch.stack(txts)
        out = [imgs, txts]

        if self.category_feats is not None:
            if len(categories) > 0:
                categories = torch.stack(categories)
            out.append(categories)

        return out
    
    def load_item(self, item):
        item_id = item['item_id']
        if self.img_feats is None:
            img_path = os.path.join(self.impath, '%s.jpg' % item_id)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = torch.tensor(self.img_feats[item_id])

        txt = torch.tensor(self.txt_feats[item_id])

        if self.category_feats is not None:
            category = torch.tensor(self.category_feats[item_id])
        else:
            category = None # or torch.zeros(1)
        return img, txt, category

    def pad_imgs_and_txts(self, imgs, txts):
        if imgs is None and txts is None:
            return None, None, None
        if len(imgs) == 0 and len(txts) == 0:
            return None, None, None
        
        if self.img_feats is None:
            T, C1, H, W = imgs.shape
        else:
            T, C1 = imgs.shape
        T, C2 = txts.shape
        assert T <= self.max_item_len, 'Too much item in outfit'

        mask = torch.zeros(self.max_item_len)
        mask[T:] = 1
        mask = mask.bool()
        if self.img_feats is None:
            padded_imgs = torch.zeros(self.max_item_len, C1, H, W)
            padded_imgs[:T, :, :, :] = imgs
        else:
            padded_imgs = torch.zeros(self.max_item_len, C1)
            padded_imgs[:T, :] = imgs
        padded_txts = torch.zeros(self.max_item_len, C2)
        padded_txts[:T, :] = txts
        
        return padded_imgs, padded_txts, mask
    
    def sample_negative_outfit(self, outfit):
        neg_outfit = {'items': [], 'set_id': outfit['set_id']}
        for item in outfit['items']:
            item_id = item['item_id']
            item_type = self.im2type[item_id]
            candidate_sets = self.category2ims[item_type].keys()

            attempts = 0
            item_out = item_id
            while item_out == item_id and attempts < 100:
                choice = random.choice(list(candidate_sets))
                choice = self.category2ims[item_type][choice]
                item_out = random.choice(choice)
                attempts += 1

            neg_outfit['items'].append({
                'item_id': item_out,
                'index': item['index']
            })
        return neg_outfit
    
    def sample_hard_negative_outfit(self, outfit):
        # sample outfit contains same items but different set
        # hard_candidates = set()
        # for item in outfit['items']:
        #     hard_candidates |= self.im2set[item['item_id']]
        # hard_candidates.remove(outfit['set_id'])
        # if len(hard_candidates) > 0:
        #     set_id = random.choice(list(hard_candidates))
        #     return self.set2data[set_id]
        # return self.sample_negative_outfit(outfit)

        # random use item from the same fine-grained category
        neg_outfit = {'items': [], 'set_id': outfit['set_id']}
        for item in outfit['items']:
            item_id = item['item_id']
            item_fg = self.im2fg[item_id]
            candidate_sets = self.fg2ims[item_fg].keys()

            attempts = 0
            item_out = item_id
            while item_out == item_id and attempts < 100:
                choice = random.choice(list(candidate_sets))
                choice = self.fg2ims[item_fg][choice]
                item_out = random.choice(choice)
                attempts += 1

            neg_outfit['items'].append({
                'item_id': item_out,
                'index': item['index']
            })
        return neg_outfit

    def __getitem__(self, index):
        outfit = deepcopy(self.data[index])

        if self.outfit_feats is not None:
            outfit_txt = self.outfit_feats[outfit['set_id']]    
        else:
            outfit_txt = torch.ones(1)
            
        imgs, txts = self.load_outfit(outfit)[:2]
        imgs, txts, mask = self.pad_imgs_and_txts(imgs, txts)

        # easy negative sampling
        negative_outfit = self.sample_negative_outfit(outfit)
        negative_imgs, negative_txts = self.load_outfit(negative_outfit)[:2]
        negative_imgs, negative_txts, negative_mask = self.pad_imgs_and_txts(negative_imgs, negative_txts)

        # hard negative sampling
        if self.sample_hard_negative:
            hard_negative_outfit = self.sample_hard_negative_outfit(outfit)
            hard_negative_imgs, hard_negative_txts = self.load_outfit(hard_negative_outfit)[:2]
            hard_negative_imgs, hard_negative_txts, hard_negative_mask = self.pad_imgs_and_txts(hard_negative_imgs, hard_negative_txts)
        else:
            hard_negative_imgs, hard_negative_txts, hard_negative_mask = torch.ones(1), torch.ones(1), torch.ones(1)
        
        return outfit_txt, imgs, txts, mask, negative_imgs, negative_txts, negative_mask, hard_negative_imgs, hard_negative_txts, hard_negative_mask

    def __len__(self):
        return len(self.data)


class UIUCPolyvoreRetrievalDataset(Dataset):
    """Dataset for retrieval, it generate partial outfits, positive item and negative item
    at the same time. It can also load FITB and retrieval question for evaluation.
    """
    def __init__(self,
            datadir=None,
            split='train',
            polyvore_split='disjoint',
            img_feat_path=None,
            txt_feat_path=None,
            category_feat_path=None,
            outfit_feat_path=None,
            max_item_len=16,
            transform=None,
            num_negative=10,
            sample_hard_negative=False
        ):
        rootdir = os.path.join(datadir, 'polyvore_outfits', polyvore_split)
        meta_data_path = os.path.join(datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
        meta_data = json.load(open(meta_data_path, 'r'))
        self.impath = os.path.join(datadir, 'polyvore_outfits', 'images')
        self.is_train = split == 'train'
        data_json = os.path.join(rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r'))
        txt_feats = pickle.load(open(txt_feat_path, 'rb'))
        if img_feat_path is not None:
            img_feats = pickle.load(open(img_feat_path, 'rb'))
        else:
            img_feats = None
        if category_feat_path is not None:
            category_feats = pickle.load(open(category_feat_path, 'rb'))
        else:
            category_feats = None
        if outfit_feat_path is not None:
            outfit_feats = pickle.load(open(outfit_feat_path, 'rb'))
        else:
            outfit_feats = None

        # get list of images and make a mapping used to quickly organize the data
        im2type = {}
        category2ims = {}
        imnames = set()
        id2im = {}
        fg2ims = {}
        im2fg = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']
                category = meta_data[im]['semantic_category']
                fg = meta_data[im]['category_id']
                im2type[im] = category
                im2fg[im] = fg

                if category not in category2ims:
                    category2ims[category] = {}
                if fg not in fg2ims:
                    fg2ims[fg] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []
                if outfit_id not in fg2ims[fg]:
                    fg2ims[fg][outfit_id] = []

                category2ims[category][outfit_id].append(im)
                fg2ims[fg][outfit_id].append(im)
                id2im['%s_%i' % (outfit_id, item['index'])] = im
                imnames.add(im)
        imnames = list(imnames)
        im2index = {}
        for index, im in enumerate(imnames):
            im2index[im] = index

        self.datadir = datadir
        self.rootdir = rootdir
        self.split = split
        
        self.data = outfit_data
        self.imnames = imnames
        self.im2type = im2type
        self.transform = transform
        self.category2ims = category2ims
        self.id2im = id2im
        self.fg2ims = fg2ims
        self.im2fg = im2fg
        self.txt_feats = txt_feats
        self.img_feats = img_feats
        self.category_feats = category_feats
        self.outfit_feats = outfit_feats
        self.max_item_len = max_item_len
        self.num_negative = num_negative
        self.sample_hard_negative = sample_hard_negative
        self.compatibility_questions = load_compatibility_questions(
            os.path.join(rootdir, f'compatibility_{split}.txt'),
                im2index,
                id2im)
        self.fitb_questions = load_fitb_questions(
            os.path.join(rootdir, f'fill_in_blank_{split}.json'),
            im2index,
            id2im)

    def sample_negative_item(self, item):
        item_id = item['item_id']
        item_type = self.im2type[item_id]
        candidate_sets = self.category2ims[item_type].keys()

        attempts = 0
        item_out = item_id
        while item_out == item_id and attempts < 100:
            choice = random.choice(list(candidate_sets))
            choice = self.category2ims[item_type][choice]
            item_out = random.choice(choice)
            attempts += 1
            
        return {'item_id': item_out}
    
    def sample_hard_negative_item(self, item):
        item_id = item['item_id']
        item_fg = self.im2fg[item_id]
        candidate_sets = self.fg2ims[item_fg].keys()

        attempts = 0
        item_out = item_id
        while item_out == item_id and attempts < 100:
            choice = random.choice(list(candidate_sets))
            choice = self.fg2ims[item_fg][choice]
            item_out = random.choice(choice)
            attempts += 1

        return {'item_id': item_out}
    
    def sample_negative_outfit(self, outfit):
        neg_outfit = {'items': [], 'set_id': outfit['set_id']}
        for item in outfit['items']:
            neg_outfit['items'].append({
                'item_id': self.sample_negative_item(item)['item_id'],
                'index': item['index']
            })
        return neg_outfit

    def load_outfit(self, outfit):
        imgs = []
        txts = []
        categories = []
        for item in outfit['items']:
            img, txt, category = self.load_item(item)
            imgs.append(img)
            txts.append(txt)
            categories.append(category)
        if self.transform is not None:
            if len(imgs) > 0:
                imgs = torch.stack(imgs)
        if len(txts) > 0:
            txts = torch.stack(txts)
        out = [imgs, txts]

        if self.category_feats is not None:
            if len(categories) > 0:
                categories = torch.stack(categories)
            out.append(categories)
        return out
    
    def load_item_img(self, item_id):
        img_path = os.path.join(self.impath, '%s.jpg' % item_id)
        img = Image.open(img_path).convert('RGB')
        return img
    
    def load_outfit_title(self, set_id):
        if not hasattr(self, 'outfit_titles'):
            self.outfit_titles = json.load(open(os.path.join(self.datadir, 'polyvore_outfits', 'polyvore_outfit_titles.json')))
        return self.outfit_titles[set_id]
    
    def load_item(self, item):
        item_id = item['item_id']
        if self.img_feats is None:
            img_path = os.path.join(self.impath, '%s.jpg' % item_id)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = torch.tensor(self.img_feats[item_id])

        txt = torch.tensor(self.txt_feats[item_id])

        if self.category_feats is not None:
            category = torch.tensor(self.category_feats[item_id])
        else:
            category = None # or torch.zeros(1)
        return img, txt, category

    def pad_imgs_and_txts(self, imgs, txts):
        if imgs is None and txts is None:
            return None, None, None
        if len(imgs) == 0 and len(txts) == 0:
            return None, None, None
        
        if self.img_feats is None:
            T, C1, H, W = imgs.shape
        else:
            T, C1 = imgs.shape
        T, C2 = txts.shape
        assert T <= self.max_item_len, 'Too much item in outfit'

        mask = torch.zeros(self.max_item_len)
        mask[T:] = 1
        mask = mask.bool()
        if self.img_feats is None:
            padded_imgs = torch.zeros(self.max_item_len, C1, H, W)
            padded_imgs[:T, :, :, :] = imgs
        else:
            padded_imgs = torch.zeros(self.max_item_len, C1)
            padded_imgs[:T, :] = imgs
        padded_txts = torch.zeros(self.max_item_len, C2)
        padded_txts[:T, :] = txts
        
        return padded_imgs, padded_txts, mask

    def __getitem__(self, index):
        outfit = deepcopy(self.data[index])

        if self.outfit_feats is not None:
            outfit_feat = self.outfit_feats[outfit['set_id']]
        else:
            outfit_feat = torch.zeros(1)

        positive = outfit['items'].pop(random.randint(0, len(outfit['items'])-1))
        positive_img, positive_txt, positive_category = self.load_item(positive)
        
        partial_imgs, partial_txts = self.load_outfit(outfit)[:2]
        partial_imgs, partial_txts, mask = self.pad_imgs_and_txts(partial_imgs, partial_txts)

        negatives = {'items': []}
        for _ in range(self.num_negative):
            negatives['items'].append(self.sample_negative_item(positive))
        negative_imgs, negative_txts = self.load_outfit(negatives)[:2]
        
        if self.sample_hard_negative:
            hard_negatives = {'items': []}
            for _ in range(self.num_negative):
                hard_negatives['items'].append(self.sample_hard_negative_item(positive))
            hard_negative_imgs, hard_negative_txts =  self.load_outfit(negatives)[:2]
        else:
            hard_negative_imgs, hard_negative_txts = torch.zeros(1), torch.zeros(1)

        return partial_imgs, partial_txts, mask, positive_img, positive_txt, positive_category, \
            negative_imgs, negative_txts, hard_negative_imgs, hard_negative_txts, outfit_feat

    def __len__(self):
        return len(self.data)
    
    def extract_emb(self, item, model, device, ctx, to_numpy=True):
        img, txt, _ = self.load_item(item)
        img, txt = img.unsqueeze(0).to(device), txt.unsqueeze(0).to(device)
        with torch.no_grad(), ctx:
            img_emb = model.img_encoder(img)
            txt_emb = model.text_encoder(txt)
            emb = torch.cat([img_emb, txt_emb], dim=-1)
        if to_numpy:
            return emb.detach().cpu().numpy()
        return emb
    
    def generate_database(self, model, device, ctx):
        """generate database for retrieval
        """
        itemId2Index = {}
        database = []
        index = 0
        for outfit in tqdm(self.data, desc='generate database'):
            for item in outfit['items']:
                if item['item_id'] not in itemId2Index:
                    itemId2Index[item['item_id']] = index
                    extract_emb = self.extract_emb(item, model, device, ctx, to_numpy=True)
                    database.append(extract_emb)
                    index += 1
        database = np.concatenate(database, axis=0)
        return database, itemId2Index


def test_predcition_dataset():
    img_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    cp_train_dat = UIUCPolyvorePredictionDataset(
        datadir='/mnt/22600F38600F1269/public_dataset/UIUC-polyvore/',
        txt_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/fashion-compatibility/encoded_title_description_distiluse-base-multilingual-cased-v2.pkl',
        img_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/compatibility-retrieval-fashionCLIP/img_feats_fashionClip.pkl',
        category_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/compatibility-retrieval-fashionCLIP/encoded_category_distiluse-base-multilingual-cased-v2.pkl',
        outfit_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/compatibility-retrieval-fashionCLIP/encoded_outfitTitle_en_fashionClip.pkl',
        transform=transform,
        split='train',
        sample_hard_negative=True
    )
    sample = cp_train_dat[0]
    outfit_txt, imgs, txts, mask, \
    negative_imgs, negative_txts, negative_mask, \
    hard_negative_imgs, hard_negative_txts, hard_negative_mask = sample


def test_retrieval_dataset():
    img_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    train_dat = UIUCPolyvoreRetrievalDataset(
        datadir='/mnt/22600F38600F1269/public_dataset/UIUC-polyvore/',
        txt_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/fashion-compatibility/encoded_title_description_distiluse-base-multilingual-cased-v2.pkl',
        img_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/compatibility-retrieval-fashionCLIP/img_feats_fashionClip.pkl',
        category_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/compatibility-retrieval-fashionCLIP/encoded_category_distiluse-base-multilingual-cased-v2.pkl',
        outfit_feat_path='/mnt/22600F38600F1269/code/sentence_bert_compatiblity/compatibility-retrieval-fashionCLIP/encoded_outfitTitle_en_fashionClip.pkl',
        transform=transform,
        split='train',
        sample_hard_negative=False
    )
    train_loader = DataLoader(train_dat, 4, shuffle=False)
    it = iter(train_loader)
    batch = next(it)
    partial_imgs, partial_txts, mask, \
    positive_img, positive_txt, positive_category, \
    negative_imgs, negative_txts, hard_negative_imgs, hard_negative_txts, \
       outfit_txt = batch
    print(positive_category.shape)
    # train_loader.dataset.sample_hard_negative = True
    # batch = next(it)


if __name__ == "__main__":
    test_predcition_dataset()
    test_retrieval_dataset()


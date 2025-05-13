import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18, vision_transformer
from einops import rearrange, repeat, reduce
from transformers import CLIPTokenizer, CLIPTextModel

from ipdb import iex, launch_ipdb_on_exception
os.environ['PYTHONBREAKPOINT'] = 'ipdb.set_trace'
    

class ResNet18ImageEncoder(nn.Module):
    def __init__(
            self,
            weights=ResNet18_Weights.DEFAULT,
            img_emb_size: int = 64,
    ):
        super(ResNet18ImageEncoder, self).__init__()
        self.img_emb_size = img_emb_size
        self.resnet18 = resnet18(weights=weights)
        self.resnet18.fc = nn.Linear(512, img_emb_size)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.resnet18.fc.weight)
        self.resnet18.fc.bias.data.fill_(0.01)

    def forward(self, x):
        return self.resnet18(x)


class VitB16ImageEncoder(nn.Module):
    def __init__(
            self,
            pretrained: bool = True,
            img_emb_size: int = 64,
    ):
        super(VitB16ImageEncoder, self).__init__()
        self.img_emb_size = img_emb_size
        self.vit_b_16 = vision_transformer.vit_b_16(pretrained=pretrained)
        self.vit_b_16.heads = nn.Sequential(
            nn.Linear(768, img_emb_size),
        )
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.vit_b_16.heads[0].weight)
        self.vit_b_16.heads[0].bias.data.fill_(0.01)

    def forward(self, x):
        return self.vit_b_16(x)


class LinearImageEncoder(nn.Module):
    def __init__(
            self,
            img_inp_size: int = 384,
            img_emb_size: int = 64,
    ):
        super(LinearImageEncoder, self).__init__()
        self.img_emb_size = img_emb_size
        self.img_inp_size = img_inp_size
        self.img_fc = nn.Linear(img_inp_size, img_emb_size)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.img_fc.weight)
        self.img_fc.bias.data.fill_(0.01)

    def forward(self, x):
        return self.img_fc(x)
    

class MlpImageEncoder(nn.Module):
    def __init__(
            self,
            img_inp_size: int = 384,
            img_emb_size: int = 64,
    ):
        super(MlpImageEncoder, self).__init__()
        self.img_emb_size = img_emb_size
        self.img_inp_size = img_inp_size
        self.img_mlp = nn.Sequential(
            nn.Linear(img_inp_size, img_inp_size),
            nn.ReLU(),
            nn.Linear(img_inp_size, img_inp_size),
            nn.ReLU(),
            nn.Linear(img_inp_size, img_emb_size),
        )
        self.init_weight()

    def init_weight(self):
        for module in self.img_mlp.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.img_mlp(x)


class LinearTextEncoder(nn.Module):
    def __init__(
            self,
            txt_inp_size: int = 384,
            txt_emb_size: int = 64,
    ):
        super(LinearTextEncoder, self).__init__()
        self.txt_inp_size = txt_inp_size
        self.txt_emb_size = txt_emb_size
        self.sentence_fc = nn.Linear(txt_inp_size, txt_emb_size)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.sentence_fc.weight)
        self.sentence_fc.bias.data.fill_(0.01)

    def forward(self, x):
        return self.sentence_fc(x)


class MlpTextEncoder(nn.Module):
    def __init__(
            self,
            txt_inp_size: int = 384,
            txt_emb_size: int = 64,
    ):
        super(MlpTextEncoder, self).__init__()
        self.txt_inp_size = txt_inp_size
        self.txt_emb_size = txt_emb_size
        self.sentence_mlp = nn.Sequential(
            nn.Linear(txt_inp_size, txt_inp_size),
            nn.ReLU(),
            nn.Linear(txt_inp_size, txt_inp_size),
            nn.ReLU(),
            nn.Linear(txt_inp_size, txt_emb_size),
        )
        self.init_weight()

    def init_weight(self):
        for module in self.sentence_mlp.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                module.bias.data.fill_(0.01)

    def forward(self, x):
        return self.sentence_mlp(x)
    

class OutfitTransformerPrediction(nn.Module):
    def __init__(
        self,
        img_encoder=LinearImageEncoder(img_inp_size=512, img_emb_size=64),
        text_encoder=LinearTextEncoder(txt_inp_size=384, txt_emb_size=64),
        outfit_txt_encoder=LinearTextEncoder(txt_inp_size=512, txt_emb_size=128),
        nhead: int = 16,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_outfit_txt: bool = True,
    ):
        super(OutfitTransformerPrediction, self).__init__()
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.outfit_txt_encoder = outfit_txt_encoder
        img_emb_size = img_encoder.img_emb_size
        txt_emb_size = text_encoder.txt_emb_size
        self.outfit_token = nn.Parameter(torch.randn(1, img_emb_size+txt_emb_size))
        self.use_outfit_txt = use_outfit_txt

        if use_outfit_txt:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=img_emb_size+txt_emb_size, nhead=nhead, batch_first=True, dropout=dropout)
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=num_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=img_emb_size+txt_emb_size, nhead=nhead, batch_first=True, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(img_emb_size+txt_emb_size, img_emb_size+txt_emb_size),
            nn.ReLU(),
            nn.Linear(img_emb_size+txt_emb_size, 1)
        )

        self.criteria = nn.BCEWithLogitsLoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.outfit_token)

        self.img_encoder.init_weight()
        self.text_encoder.init_weight()

        nn.init.xavier_normal_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0.01)
        
    def get_loss(self, logits, target):
        loss = self.criteria(logits, target)
        return loss
    
    def extract_outfit_feats(self, imgs, txts):
        # encoding existing fashion items
        imgs_emb = self.img_encoder(rearrange(imgs, 'N S ... -> (N S) ...'))
        txts_emb = self.txt = self.text_encoder(rearrange(txts, 'N S ... -> (N S) ...'))
        outfit_emb = torch.cat([imgs_emb, txts_emb], dim=-1)
        outfit_emb = rearrange(outfit_emb, '(N S) ... -> N S ...', N=imgs.shape[0])
        return outfit_emb
    
    def forward(
            self,
            imgs: torch.Tensor, 
            txts: torch.Tensor, 
            mask: Optional[torch.Tensor]=None, 
            outfit_txt: Optional[torch.Tensor]=None,
            target: Optional[torch.Tensor]=None,
            return_feats: bool=False,
        ):
        N = imgs.shape[0]

        outfit_emb = self.extract_outfit_feats(imgs, txts)

        outfit_token = self.outfit_token.expand(N, -1)

        # encode outfit text
        if outfit_txt is not None:
            outfit_txt_emb = self.outfit_txt_encoder(outfit_txt)[:, None, :]
            memory = torch.cat([outfit_txt_emb, outfit_emb], dim=1)

        # put target_item_token at the beginning of sequence
        outfit_emb = torch.cat([
            outfit_token.unsqueeze(1),
            outfit_emb
        ], dim=1)

        # make mask for transformer encoder
        if mask is not None:
            mask = torch.cat([
                torch.zeros(N, 1).bool().to(mask.device),
                mask
            ], dim=1)
            if self.use_outfit_txt:
                logits = self.transformer_decoder(outfit_emb, memory, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
            else:
                logits = self.transformer_encoder(outfit_emb, src_key_padding_mask=mask)
        else:
            if self.use_outfit_txt:
                logits = self.transformer_decoder(outfit_emb)
            else:
                logits = self.transformer_encoder(outfit_emb)

        logits = logits[:, 0]
        
        if return_feats:
            return {'feats': logits}

        logits = self.mlp(logits)

        if target is not None:
            loss = self.get_loss(logits, target)
            return {'logits': logits, 'loss': loss}

        return {'logits': logits}


class OutfitTransformerRetrieval(nn.Module):
    def __init__(
        self,
        img_encoder=LinearImageEncoder(img_inp_size=512, img_emb_size=64),
        text_encoder=LinearTextEncoder(txt_inp_size=384, txt_emb_size=64),
        outfit_txt_encoder=LinearTextEncoder(txt_inp_size=512, txt_emb_size=128),
        nhead: int = 16,
        num_layers: int = 6,
        margin: float = 0.3,
        target_item_info: Optional[str] = None, # 'category', 'text', or None
        dropout: float = 0.1,
        use_outfit_txt: bool = True,
    ):
        super(OutfitTransformerRetrieval, self).__init__()
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.outfit_txt_encoder = outfit_txt_encoder
        img_emb_size = img_encoder.img_emb_size
        txt_emb_size = text_encoder.txt_emb_size
        self.target_item_info = target_item_info
        self.use_outfit_txt = use_outfit_txt

        if target_item_info is None:
            self.target_img_token = nn.Parameter(torch.randn(1, img_emb_size+txt_emb_size))
        else:
            self.target_img_token = nn.Parameter(torch.randn(1, img_emb_size))

        if self.use_outfit_txt:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=img_emb_size+txt_emb_size, nhead=nhead, batch_first=True, dropout=dropout)
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=num_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=img_emb_size+txt_emb_size, nhead=nhead, batch_first=True, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(img_emb_size+txt_emb_size, img_emb_size+txt_emb_size),
            nn.ReLU(),
            nn.Linear(img_emb_size+txt_emb_size, img_emb_size+txt_emb_size)
        )

        self.ranking_loss = nn.MarginRankingLoss(margin)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.target_img_token)

        self.img_encoder.init_weight()
        self.text_encoder.init_weight()

        nn.init.xavier_normal_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0.01)

    def from_cp_pretriained(self, pretrained_model_path: str):
        """load pretrained compatibility prediction checkpoint
        """
        state_dict = torch.load(pretrained_model_path)
        self.load_state_dict(state_dict)

    def get_loss(self, logits, positive_emb, negative_emb, hard_negative_emb=None):
        N, num_negative = negative_emb.shape[:2]
        ranking_target = torch.ones(N, num_negative).to(positive_emb.device)
        D_p = F.cosine_similarity(logits, positive_emb, -1)
        D_n = F.cosine_similarity(logits.unsqueeze(1), negative_emb, -1)
        loss = self.ranking_loss(D_p.unsqueeze(-1), D_n, ranking_target)

        if hard_negative_emb is not None:
            D_hn = F.cosine_similarity(logits.unsqueeze(1), hard_negative_emb, -1)
            D_hn = torch.max(D_hn, dim=-1)[0]
            ranking_target_hard = torch.ones(N).to(positive_emb.device)
            loss_hard = self.ranking_loss(D_p, D_hn, ranking_target_hard)
            loss += loss_hard
        return loss
    
    def forward_v2(
            self,
            partial_imgs: torch.Tensor, 
            partial_txts: torch.Tensor, 
            mask: Optional[torch.Tensor]=None, 
            positive_img: Optional[torch.Tensor]=None, 
            positive_txt: Optional[torch.Tensor]=None,
            positive_category: Optional[torch.Tensor]=None, 
            negative_imgs: Optional[torch.Tensor]=None, 
            negative_txts: Optional[torch.Tensor]=None,
            hard_negative_imgs: Optional[torch.Tensor]=None, 
            hard_negative_txts: Optional[torch.Tensor]=None,
            outfit_txt: Optional[torch.Tensor]=None
        ) -> dict:
        N, S = mask.shape

        # encode partial items
        if partial_imgs is None and partial_txts is None:
            partial_emb = None
        else:
            partial_imgs_emb = self.img_encoder(
                rearrange(partial_imgs, 'n s ... -> (n s) ...'))
            partial_imgs_emb = rearrange(
                partial_imgs_emb, '(n s) ... -> n s ...', n=N)
            partial_txts_emb = self.text_encoder(partial_txts)
            partial_emb = torch.cat([partial_imgs_emb, partial_txts_emb], dim=-1)

        # make target item token
        if self.target_item_info is None:
            target_item_token = self.target_img_token.expand(N, -1)
        elif self.target_item_info == 'text' and positive_txt is not None:
            positive_txt_emb = self.text_encoder(positive_txt)
            target_item_token = torch.concat(
                [self.target_img_token.expand(N, -1), positive_txt_emb], dim=1)
        elif self.target_item_info == 'category' and positive_category is not None:
            positive_category_emb = self.text_encoder(positive_category)
            target_item_token = torch.concat(
                [self.target_img_token.expand(N, -1), positive_category_emb], dim=1)
            
        # encode outfit text
        if self.use_outfit_txt and outfit_txt is not None:
            outfit_emb = self.outfit_txt_encoder(outfit_txt)[:, None, :]
            if partial_emb is None:
                memory = outfit_emb
            else:
                memory = torch.cat([outfit_emb, partial_emb], dim=1)

        # put target_item_token at the beginning of sequence
        if partial_emb is None:
            partial_emb = target_item_token.unsqueeze(1)
        else:
            partial_emb = torch.cat([
                target_item_token.unsqueeze(1),
                partial_emb
            ], dim=1)

        # make mask for transformer encoder
        if mask is not None:
            mask = torch.cat([
                torch.zeros(N, 1).bool().to(mask.device),
                mask
            ], dim=1)
            if self.use_outfit_txt:
                logits = self.transformer_decoder(partial_emb, memory, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
            else:
                logits = self.transformer_encoder(partial_emb, src_key_padding_mask=mask)
        else:
            if self.use_outfit_txt:
                logits = self.transformer_decoder(partial_emb, memory)
            else:
                logits = self.transformer_encoder(partial_emb)

        logits = logits[:, 0]

        logits = self.mlp(logits)

        # compute loss if target exists
        if (positive_img is not None and
            positive_txt is not None and
            negative_imgs is not None and
            negative_txts is not None):
            
            # encode negative items
            negative_imgs_emb = self.img_encoder(
                rearrange(negative_imgs, 'n neg ... -> (n neg) ...'))
            negative_imgs_emb = rearrange(
                negative_imgs_emb, '(n neg) ... -> n neg ...', n=N)
            negative_txts_emb = self.text_encoder(negative_txts)
            negative_emb = torch.cat([negative_imgs_emb, negative_txts_emb], dim=-1)

            # encode negative items
            if hard_negative_imgs is not None and hard_negative_txts is not None:
                hard_negative_imgs_emb = self.img_encoder(
                    rearrange(hard_negative_imgs, 'n neg ... -> (n neg) ...'))
                hard_negative_imgs_emb = rearrange(
                    hard_negative_imgs_emb, '(n neg) ... -> n neg ...', n=N)
                hard_negative_txts_emb = self.text_encoder(hard_negative_txts)
                hard_negative_emb = torch.cat([hard_negative_imgs_emb, hard_negative_txts_emb], dim=-1)
            else:
                hard_negative_emb = None

            # encode positive items
            positive_img_emb = self.img_encoder(positive_img)
            if self.target_item_info != 'text':
                positive_txt_emb = self.text_encoder(positive_txt)
            positive_emb = torch.concat(
                [positive_img_emb, positive_txt_emb], dim=-1)
         
            # compute ranking loss
            loss = self.get_loss(logits, positive_emb, negative_emb, hard_negative_emb)
            
            return {
                'loss': loss,
                'logits': logits,
            }
        
        return {
            'logits': logits
        }

    def forward(
            self,
            partial_imgs: torch.Tensor, 
            partial_txts: torch.Tensor, 
            mask: Optional[torch.Tensor]=None, 
            positive_img: Optional[torch.Tensor]=None, 
            positive_txt: Optional[torch.Tensor]=None,
            positive_category: Optional[torch.Tensor]=None, 
            negative_imgs: Optional[torch.Tensor]=None, 
            negative_txts: Optional[torch.Tensor]=None,
            hard_negative_imgs: Optional[torch.Tensor]=None, 
            hard_negative_txts: Optional[torch.Tensor]=None,
            outfit_txt: Optional[torch.Tensor]=None
        ) -> dict:
        if negative_txts is not None:
            _, num_negative, _ = negative_txts.shape

        # encode partial items
        if partial_imgs is None and partial_txts is None:
            partial_emb = None
            N, _ = positive_category.shape
        else:
            if len(partial_imgs.shape) == 5:
                N, S, C1, H, W = partial_imgs.shape
                _, _, C2 = partial_txts.shape
                partial_imgs_emb = self.img_encoder(partial_imgs.reshape(-1, C1, H, W))
                partial_txts_emb = self.text_encoder(partial_txts.reshape(-1, C2))
            elif len(partial_imgs.shape) == 3:
                N, S, C1 = partial_imgs.shape
                _, _, C2 = partial_txts.shape

                partial_imgs_emb = self.img_encoder(partial_imgs)
                partial_txts_emb = self.text_encoder(partial_txts)
            partial_emb = torch.cat([partial_imgs_emb, partial_txts_emb], dim=-1)
            partial_emb = partial_emb.reshape(N, S, -1)

        # make target item token
        if self.target_item_info is None:
            target_item_token = self.target_img_token.expand(N, -1)
        elif self.target_item_info == 'text' and positive_txt is not None:
            positive_txt_emb = self.text_encoder(positive_txt)
            target_item_token = torch.concat(
                [self.target_img_token.expand(N, -1), positive_txt_emb], dim=1)
        elif self.target_item_info == 'category' and positive_category is not None:
            positive_category_emb = self.text_encoder(positive_category)
            target_item_token = torch.concat(
                [self.target_img_token.expand(N, -1), positive_category_emb], dim=1)
            
        # encode outfit text
        if self.use_outfit_txt and outfit_txt is not None:
            outfit_emb = self.outfit_txt_encoder(outfit_txt)[:, None, :]
            if partial_emb is None:
                memory = outfit_emb
            else:
                memory = torch.cat([outfit_emb, partial_emb], dim=1)

        # put target_item_token at the beginning of sequence
        if partial_emb is None:
            partial_emb = target_item_token.unsqueeze(1)
        else:
            partial_emb = torch.cat([
                target_item_token.unsqueeze(1),
                partial_emb
            ], dim=1)

        # make mask for transformer encoder
        if mask is not None:
            mask = torch.cat([
                torch.zeros(N, 1).bool().to(mask.device),
                mask
            ], dim=1)
            if self.use_outfit_txt:
                logits = self.transformer_decoder(partial_emb, memory, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
            else:
                logits = self.transformer_encoder(partial_emb, src_key_padding_mask=mask)
        else:
            if self.use_outfit_txt:
                logits = self.transformer_decoder(partial_emb, memory)
            else:
                logits = self.transformer_encoder(partial_emb)

        logits = logits[:, 0]

        logits = self.mlp(logits)

        # compute loss if target exists
        if (positive_img is not None) and \
            (positive_txt is not None) and \
            (negative_imgs is not None) and \
            (negative_txts is not None):
            
            # encode negative items
            if len(negative_imgs.shape) == 4:
                negative_imgs_emb = self.img_encoder(
                    negative_imgs.reshape(-1, C1, H, W))
            elif len(negative_imgs.shape) == 3:
                negative_imgs_emb = self.img_encoder(
                    negative_imgs.reshape(-1, C1))
            negative_txts_emb = self.text_encoder(negative_txts.reshape(-1, C2))
            negative_emb = torch.cat([negative_imgs_emb, negative_txts_emb], dim=-1)
            negative_emb = negative_emb.reshape(N, num_negative, -1)

            # encode negative items
            if hard_negative_imgs is not None and hard_negative_txts is not None:
                if len(hard_negative_imgs.shape) == 4:
                    hard_negative_imgs_emb = self.img_encoder(
                        hard_negative_imgs.reshape(-1, C1, H, W))
                elif len(hard_negative_imgs.shape) == 3:
                    hard_negative_imgs_emb = self.img_encoder(
                        hard_negative_imgs.reshape(-1, C1))
                hard_negative_txts_emb = self.text_encoder(hard_negative_txts.reshape(-1, C2))
                hard_negative_emb = torch.cat([hard_negative_imgs_emb, hard_negative_txts_emb], dim=-1)
                hard_negative_emb = hard_negative_emb.reshape(N, num_negative, -1)
            else:
                hard_negative_emb = None

            # encode positive items
            positive_img_emb = self.img_encoder(positive_img)
            if self.target_item_info != 'text':
                positive_txt_emb = self.text_encoder(positive_txt)
            positive_emb = torch.concat(
                [positive_img_emb, positive_txt_emb], dim=-1)
         
            # compute ranking loss
            loss = self.get_loss(logits, positive_emb, negative_emb, hard_negative_emb)
            # ranking_target = torch.ones(N, num_negative).to(positive_emb.device)
            # D_p = F.cosine_similarity(logits, positive_emb, -1)
            # D_n = F.cosine_similarity(logits.unsqueeze(1), negative_emb, -1)
            # loss = self.ranking_loss(D_p.unsqueeze(-1), D_n, ranking_target)
            
            return {
                'loss': loss,
                'logits': logits,
            }
        
        return {'logits': logits}


@iex
def test_OutfitTransformerPrediction():
    torch.random.manual_seed(0)
    imgs = torch.randn(4, 16, 512)
    txts = torch.randn(4, 16, 384)
    mask = torch.zeros(4, 16).bool()
    outfit_txt = torch.randn(4, 512)
    target = torch.ones(4, 1)
    m = OutfitTransformerPrediction(
        img_encoder=LinearImageEncoder(512, 64),
        text_encoder=LinearTextEncoder(384, 64),
        outfit_txt_encoder=LinearTextEncoder(512, 128),
    )
    m.eval()
    out = m(imgs, txts, mask, outfit_txt, target)
    print(out['logits'])
    print(out['loss'])


@iex
def test_OutfitTransformerRetrieval():
    torch.random.manual_seed(0)

    partial_imgs = torch.randn(4, 16, 512)
    partial_txts = torch.randn(4, 16, 384)
    mask = torch.zeros(4, 16).bool()
    positive_txt = torch.randn(4, 384)
    positive_img = torch.randn(4, 512)
    positive_category = torch.randn(4, 384)
    negative_imgs = torch.randn(4, 10, 512)
    negative_txts = torch.randn(4, 10, 384)
    outfit_txt = torch.randn(4, 512)
    m = OutfitTransformerRetrieval(
        img_encoder=LinearImageEncoder(512, 64),
        text_encoder=LinearTextEncoder(384, 64),
        outfit_txt_encoder=LinearTextEncoder(512, 128),
        target_item_info='category'
    )
    m.eval()
    out = m(
        partial_imgs, partial_txts, mask, 
        positive_img, positive_txt, positive_category, 
        negative_imgs, negative_txts, 
        hard_negative_imgs=None, hard_negative_txts=None,
        outfit_txt=outfit_txt
    )
    out2 = m(
        partial_imgs, partial_txts, None, 
        positive_img, positive_txt, positive_category, 
        negative_imgs, negative_txts, 
        hard_negative_imgs=None, hard_negative_txts=None,
        outfit_txt=outfit_txt
    )
    print(torch.allclose(out['logits'], out2['logits']))


if __name__ == "__main__":
    test_OutfitTransformerRetrieval()
    test_OutfitTransformerPrediction()
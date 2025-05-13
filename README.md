# Text-Conditioned Outfit Recommendation

This respository contains source code for the paper [Text-Conditioned Outfit Recommendation With Hybrid Attention Layer](https://ieeexplore.ieee.org/document/10373836). 

The model in this paper aim to recommend a whole set of fashion items with internal compatibility and text description coherence.

![task](assets/task.jpg)

## Data Preparation

The dataset used in this paper is from [UIUC Polyvore](https://github.com/mvasil/fashion-compatibility). You can download the dataset from [this line](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing)

First use pretrained model to extract features from item images, item text, outfit text. You can download extracted features or extract by yourself. If download extracted features, put these file into project directory:

- [imgs_feats_fashionClip.pkl](https://drive.google.com/drive/folders/1VwbsS_fTduYENW30vdFGPwoie9Faxlzn?usp=sharing): item image features
- [encoded_itemTitleDescription_distiluse-base-multilingual-cased-v2.pkl](https://drive.google.com/drive/folders/1VwbsS_fTduYENW30vdFGPwoie9Faxlzn?usp=sharing): item text features
- [encoded_category_distiluse-base-multilingual-cased-v2.pkl](https://drive.google.com/drive/folders/1VwbsS_fTduYENW30vdFGPwoie9Faxlzn?usp=sharing): item category features
- [encoded_outfitUrlTitle_en_fashionClip.pkl](https://drive.google.com/drive/folders/1VwbsS_fTduYENW30vdFGPwoie9Faxlzn?usp=sharing): outfit text features

If you extract features by yourself, The following scripts may help you:

```bash
# Prepare extracted image, text, category features of items
python scripts/extract_item_image_with_fashionClip.py
python scripts/extract_item_txt_with_sentenceBERT.py
python scripts/extract_item_category_with_sentenceBERT.py

# Prepare text features of outfit text
python scripts/translate_outfit_txt.py # translate to english
python scripts/extract_outfit_txt_fashionClip.py
```

## Train

Train with compatibility prediction task:

```bash
python train_cp.py
```

Train with CIR task:

```bash
python train_cir.py
```

## Evaluate

To evaluate compatibility prediction task, use the following script:

```bash
python evaluate_cp.py
```

To evaluate CIR task, use the following script:

```bash
python evaluate_cir.py
```

pretrained model weights trained on UIUC Polyvore can be found at [here](https://drive.google.com/drive/folders/1A5t3NTArQjGpJLmSPBghONaPJ9rH8AbT?usp=sharing).
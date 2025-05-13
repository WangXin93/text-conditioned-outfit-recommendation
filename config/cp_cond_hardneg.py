from .base_config import *

# model config
use_outfit_txt=True

# I/O config
out_dir = Path('runs/outfit_transformer_prediction_itemTitleDescription_outfitUrlTitle_cond_hardneg')
if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)
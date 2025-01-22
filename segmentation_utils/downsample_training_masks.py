from pathlib import Path
import PIL
from PIL import Image
import os

# downsample_width = 3840//2
# downsample_height = 2160//2

upsample_width = 3840
upsample_height = 2160

# out_path = Path('~/wood_fourth_temp').expanduser()
out_path = Path('~/someframes_preannotations').expanduser()
out_path.mkdir(parents=True, exist_ok=True)

# in_path = Path('~/wood_light_normal_temp').expanduser()
in_path = Path('~/someframes_preannotations_bw').expanduser()

for train_file in os.listdir(in_path):
    print(train_file)
    if train_file.endswith(".png") or train_file.endswith(".jpg"):
        img_pil = PIL.Image.open(os.path.join(in_path, train_file))
        # smaller_pil = img_pil.resize((downsample_width, downsample_height), Image.ANTIALIAS)
        smaller_pil = img_pil.resize((upsample_width, upsample_height))
        smaller_pil.save(os.path.join(out_path, train_file))
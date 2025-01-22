from pathlib import Path
import PIL
from PIL import Image
import os

downsample_width = 960
downsample_height = 540

out_path = Path('~/r/final_gsw1/nonfloor_segmentation_downsampled_bw').expanduser()
out_path.mkdir(parents=True, exist_ok=True)

in_path = Path('~/r/final_gsw1/nonfloor_segmentation').expanduser()

for train_file in os.listdir(in_path):
    print(train_file)
    # if not train_file.endswith("inbounds.png"):
    #     img_pil = PIL.Image.open(train_file).convert("RGB")
    # else:
    #     img_pil = PIL.Image.open(train_file)
    if train_file.endswith(".png"):
        if "color" in train_file:
            img_pil = PIL.Image.open(os.path.join(in_path, train_file)).convert('LA')
        else:
            img_pil = PIL.Image.open(os.path.join(in_path, train_file))
        smaller_pil = img_pil.resize((downsample_width, downsample_height), Image.ANTIALIAS)
        smaller_pil.save(os.path.join(out_path, train_file))
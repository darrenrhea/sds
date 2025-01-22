from UNet1 import UNet1
from pathlib import Path
from stride_score_image import stride_score_image
import time
import PIL
import PIL.Image
import numpy as np

from pathlib import Path
import pprint as pp
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import my_pytorch_utils
from get_the_torch_device_and_model import get_the_torch_device_and_model



torch_device, model = get_the_torch_device_and_model()

threshold = 0.5
input_extension = "jpg" # bmp is faster to read and write, but huge
video_name = "swinney1"
masking_attempt_id = "training_on_6"
save_color_information_into_masks = False
# height = 1080
# width = 1920

# does not work:
i_min = 580
i_max = 804
j_min = 600
j_max = 1048



i_min = 652
i_max =876
j_min = 843
j_max = 1067
# works:

# i_min = 0
# i_max = 1080
# j_min = 0
# j_max = 1920




height = i_max - i_min
width = j_max - j_min
stride = 224

masking_attempts_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts").expanduser()
masking_attempts_dir.mkdir(exist_ok=True)
out_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts/{masking_attempt_id}").expanduser()
out_dir.mkdir(exist_ok=True)
start_time = time.time()
num_images_scored = 0
for frame_index in range(4601, 4601+1):
    color_original_pngs_dir = Path(
        f"~/awecom/data/clips/{video_name}/frames").expanduser()
    image_path = color_original_pngs_dir / f"{video_name}_{frame_index:06d}.{input_extension}"
    cropped_image_path = color_original_pngs_dir / f"{video_name}_{frame_index:06d}_cropped_{j_min}_{j_max}_{i_min}_{i_max}.{input_extension}"
    assert image_path.is_file(), f"{image_path} does not exist!"
    img_pil = PIL.Image.open(str(image_path)).convert("RGB")
    hwc_np_uint8 = np.array(img_pil)
    
    binary_prediction = stride_score_image(
        hwc_np_uint8=hwc_np_uint8,
        j_min=j_min,
        i_min=i_min,
        j_max=j_max,
        i_max=i_max,
        torch_device=torch_device,
        model=model,
        threshold=threshold,
        stride=stride,
        batch_size=64
    )
    print(f"binary prediction {binary_prediction.shape}")

    stop_time = time.time()
    num_images_scored += 1
    print(f"Took {stop_time - start_time} to score {num_images_scored} images:\n{image_path}")
    print(f"Took {(stop_time - start_time) / (num_images_scored)} seconds per image, or {(num_images_scored) / (stop_time - start_time)} images per second")


    if (save_color_information_into_masks):
        out_hwc_rgba_uint8 = np.zeros(shape=(height, width, 4), dtype=np.uint8)
        out_hwc_rgba_uint8[:, :, :3] = hwc_np_uint8
        out_hwc_rgba_uint8[:, :, 3] = binary_prediction * 255
        out_pil = PIL.Image.fromarray(out_hwc_rgba_uint8)
        # save it to file:

        out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
        out_pil.save(out_file_name, "PNG")
    else:
        out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
        out_hw_grayscale_uint8[:, :] = binary_prediction * 255
        out_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
        # save it to file:
        out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor_{stride}.png"
        out_pil.save(out_file_name, "PNG")
    
    img_crop = img_pil.crop((j_min, i_min, j_max, i_max))
    img_crop.save(cropped_image_path)
    print(f"cropped color image is at:\npri {cropped_image_path}")
        
    print(f"pri {out_file_name}")

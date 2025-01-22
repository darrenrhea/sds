"""
This might be too binary/hard-decisiony.  Make feathered or at least have the option.
"""
import sys
from stitch_togetherer import stitch_togetherer
import numpy as np
import torch
import torch.nn.functional as F
import time
from fastai.vision.all import *
import PIL
import PIL.Image
import my_pytorch_utils
from pathlib import Path
import numpy as np
import better_json
import sys
import pprint as pp

clip_id = "BOS_CORE_2022-04-03_WSH_youtube"
# clip_id = "BOS_CORE_2022-03-30_MIA_PGM_30Mbps"
# clip_id = "BOS_CORE_2022-03-13_DAL_youtube"

input_dir = Path(
    f"~/awecom/data/clips/{clip_id}/frames/"
).expanduser()

output_dir = Path(
    "~/crazybake"
).expanduser()


base_names = [
    f"{clip_id}_{frame_index:06d}.jpg"
    for frame_index in range(0, 30000+1, 1000)
]

input_path_output_path_pairs = [
    (
        input_dir / base_name,
        output_dir / f"{base_name[:-4]}_nonfloor.png" 
    )
    for base_name in base_names
]

pp.pprint(input_path_output_path_pairs)


config_dict = {
    "video_name": "BKN_CITY_2022-01-03_PGM_short",
    "first_frame_index": 3820,
    "last_frame_index": 3820,
    "which_gpu": 1,
    "architecture": "resnet34",
    "nn_input_width": 400,
    "nn_input_height": 400,
    "photo_width": 1920,
    "photo_height": 1080,
    "model_name": "brooklyn_400p_400p_res34_32e_88f_full_res",  #  "celtics_400p_400p_res34_1e_3f_crazybake_full_res"
    "masking_attempt_id": "brooklyn_400p_400p_res34_32e_88f_full_res",
    "increment_frame_index_by": 1,
    "save_color_information_into_masks": True
}

video_name = config_dict['video_name']
first_frame_index = config_dict['first_frame_index']
last_frame_index = config_dict['last_frame_index']
which_gpu = config_dict['which_gpu']
model_name = config_dict['model_name']
masking_attempt_id = config_dict['masking_attempt_id']
increment_frame_index_by = config_dict['increment_frame_index_by']
save_color_information_into_masks = config_dict['save_color_information_into_masks']
architecture = config_dict['architecture']
nn_input_width = config_dict['nn_input_width']
nn_input_height = config_dict['nn_input_height']
photo_width = config_dict['photo_width']
photo_height = config_dict['photo_height']

assert last_frame_index >= first_frame_index
assert which_gpu in [0, 1, 2, 3], "gpu id must be 0, 1, 2, or 3, see nvidia-smi"


# path = Path('~/r/gsw1/led_224_224_croppings').expanduser()
croppings_path = Path('~/brooklyn_nets_barclays_center_croppings/400x400_full_res').expanduser()
croppings_path.mkdir(exist_ok=True)
model_path = Path("~/r/trained_models").expanduser()

torch_device = my_pytorch_utils.get_the_correct_gpu("8000", which_copy=which_gpu)
fnames = list(croppings_path.glob('*_color.png')) # list of input frame paths
codes = np.array(["foreground", "background"])
def label_func(fn):
    return croppings_path/f"{fn.stem[:-6]}_nonfloor{fn.suffix}"
dls = SegmentationDataLoaders.from_label_func(
    croppings_path, 
    bs=16, 
    fnames = fnames, 
    label_func = label_func, 
    codes = codes,
    valid_pct=0.1,
    seed=42, # random seed,
    batch_tfms=aug_transforms(
        mult=1.0,
        do_flip=True, 
        flip_vert=False, 
        max_rotate=10.0, 
        min_zoom=1.0, 
        max_zoom=1.1, 
        max_lighting=0.2, 
        max_warp=0.2, 
        p_affine=0.75, 
        p_lighting=0.75, 
        xtra_tfms=None, 
        size=None, 
        mode='bilinear', 
        pad_mode='reflection', 
        align_corners=True, 
        batch=False, 
        min_scale=1.0)
)
# architecture = sys.argv[7]
if architecture == "resnet34":
    arch = resnet34
    print(f"arch is {architecture}")
elif architecture == "resnet18":
    arch = resnet18
learner = unet_learner(dls=dls, arch=arch)
print(f"instantiated learner")
model = learner.load(model_path/model_name)
model = model.to(torch_device)
torch.cuda.synchronize()
# model.eval()
threshold = 0.5
input_extension = "jpg" # bmp is faster to read and write, but huge

i_min = 0
j_min = 0
j_max = photo_width
i_max = photo_height
width = j_max - j_min
height = i_max - i_min

masking_attempts_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts").expanduser()
masking_attempts_dir.mkdir(exist_ok=True)
out_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts/{masking_attempt_id}").expanduser()
out_dir.mkdir(exist_ok=True)
start_time = time.time()
num_images_scored = 0
for image_path, output_path in input_path_output_path_pairs:
    if not image_path.is_file():
        print(f"{image_path} does not exist")
        sys.exit(1)

    img_pil = PIL.Image.open(str(image_path)).convert("RGB")
    hwc_np_uint8 = np.array(img_pil)
    
    binary_prediction = stitch_togetherer(
        hwc_np_uint8=hwc_np_uint8,
        i_min=i_min,
        i_max=i_max,
        j_min=j_min,
        j_max=j_max,
        torch_device=torch_device,
        model=model,
        nn_input_width=nn_input_width,
        nn_input_height=nn_input_height,
        threshold=threshold,
        stride=224,
        batch_size=45
    )
    

    num_images_scored += 1

    score_image = PIL.Image.fromarray(
        np.clip(binary_prediction * 255.0, 0, 255).astype(np.uint8))

    if (save_color_information_into_masks):
        out_hwc_rgba_uint8 = np.zeros(shape=(height, width, 4), dtype=np.uint8)
        out_hwc_rgba_uint8[:, :, :3] = hwc_np_uint8
        out_hwc_rgba_uint8[:, :, 3] = binary_prediction * 255
        out_pil = PIL.Image.fromarray(out_hwc_rgba_uint8)
        # save it to file:

        out_file_name = str(output_path)
        # even full color goes fast if you turn down compression level:
        png_save_start = time.time()
        out_pil.save(out_file_name, "PNG", optimize=False, compress_level=0)
        png_save_stop = time.time()
        print(f"Took {png_save_stop - png_save_start} seconds to save png")
    else:
        out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
        out_hw_grayscale_uint8[:, :] = binary_prediction * 255
        out_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
        # save it to file:
        out_pil.save(out_file_name, "PNG")  # fast because it is black white
        
    print(f"pri {out_file_name}")

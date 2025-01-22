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

"""
This has been changed to take in a single JSON config file that is something like:

{
    "video_name": "den1",
    "first_frame_index": 147000,
    "last_frame_index": 185000,
    "gpu_index": 0,
    "architecture": "resnet34",
    "nn_input_width": 224,
    "nn_input_height": 224,
    "model_name": "den1_fastai_224p_224p_res34_32e_10f",
    "masking_attempt_id": "fastai_den1_resnet34_224_video",
    "increment_frame_index_by": 500,
    "save_color_information_into_masks": false
}

conda activate floor_not_floor
python run_stitcher.py fastai_gsw1_program_feed.json
"""

# time python run_stitcher.py gsw1 147000 347000 0 fastai_224p_res34_32e_10f fastai_resnet34_224 resnet34 > out34_16.txt
# time python run_stitcher.py gsw1 147000 347000 0 fastai_224p_res34_32e_10f fastai_resnet34_224 resnet34 > out34_16.txt
# time python run_stitcher.py den1 146000 651000 0 den1_fastai_224p_224p_res34_32e_10f fastai_den1_resnet34_224 resnet34 > out_den1.txt
# time python run_stitcher.py den1 147000 148000 0 den1_fastai_224p_224p_res34_32e_10f fastai_den1_resnet34_224_video resnet34 > out_den1_video.txt
# time python run_stitcher.py den1 147000 148000 0 den1_gsw1_fastai_224p_224p_res34_30e_10f fastai_den1_gsw1_resnet34_224 resnet34 > out_den1_gsw1.txt
# time python run_stitcher.py ind1 300000 310000 0 den1_gsw1_ind1_fastai_224p_224p_res34_32e_10f fastai_den1_gsw1_ind1_resnet34_224_just_ind1 resnet34 > out_combined_ind1.txt
# time python run_stitcher.py den1 147000 300000 0 den1_gsw1_ind1_fastai_224p_224p_res34_32e_10f fastai_den1_gsw1_ind1_resnet34_224_just_den1 resnet34 > out_combined_den1.txt
# time python run_stitcher.py gsw1 147000 150000 0 den1_gsw1_ind1_fastai_224p_224p_res34_32e_10f fastai_den1_gsw1_ind1_resnet34_224_just_gsw1 resnet34 > out_combined_gsw1.txt


# doing gsw1 in 4 parts:

# loki Titan V doing this:
# time python run_stitcher.py gsw1 0 51500 0 final_bw

# loki Tesla V100-32GB doing this:
# time python run_stitcher.py gsw1 51501 103000 1 final_bw

# lam RTX 0 doing this:
# time python run_stitcher.py gsw1 103001 154500 0 final_bw

# lam RTX 1 doing this:
# time python run_stitcher.py gsw1 154501 204755 1 final_bw

json_config_path = sys.argv[1]
json_config_file = Path(json_config_path).expanduser()
config_dict = better_json.load(json_config_file)
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

# assert video_name in [
#     "gsw1",
#     "curry",
#     "den1",
#     "ind1",
#     "okst1",
#     "okstfull",
#     "gsw1_multi",
#     "lakers1",
#     "nets1",
#     "nets1progfeed",
#     "nets2",
#     "nets2progfeed",
#     "nets20211103",
#     "nets20211116",
#     "nets20211117",
#     "nets20211127",
#     "nets20211130",
#     "nets20211203",
#     "nets20211204",
# ]
assert last_frame_index >= first_frame_index
assert which_gpu in [0, 1, 2, 3], "gpu id must be 0, 1, 2, or 3, see nvidia-smi"

if False:
    assert model_name in [
        "fastai_27e_50f",
        "fastai_24e_66f",
        "fastai_26e_66f",
        "fastai_29e_66f",
        "fastai_30e_66f",
        "fastai_resnet18_8e_10f",
        "fastai_resnet18_224p_32e_10f",
        "fastai_224p_res34_32e_10f",
        "den1_fastai_224p_224p_res34_32e_10f",
        "den1_gsw1_fastai_224p_224p_res34_30e_10f",
        "den1_gsw1_fastai_224p_224p_res34_32e_10f",
        "den1_gsw1_ind1_fastai_224p_224p_res34_32e_10f",
        "den1_gsw1_ind1_okstfull_fastai_224p_224p_res34_32e_34f",
        "okstfull_fastai_224p_224p_res34_32e_14f",
        "fastai_led_224p_224p_res34_31e_18f",
        "fastai_gsw1_scorebug_224p_224p_res34_1e_71f",
        "fastai_brooklyn_224p_224p_res34_32e_6f"
    ]


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
for frame_index in range(first_frame_index, last_frame_index + 1, increment_frame_index_by):
    color_original_pngs_dir = Path(
        f"~/awecom/data/clips/{video_name}/frames").expanduser()
    image_path = color_original_pngs_dir / f"{video_name}_{frame_index:06d}.{input_extension}"
    # cropped_image_path = color_original_pngs_dir / f"{video_name}_{frame_index:06d}_cropped.{input_extension}"
    if not image_path.is_file():
        continue
    # assert image_path.is_file(), f"{image_path} does not exist!"
    pil_start = time.time()
    img_pil = PIL.Image.open(str(image_path)).convert("RGB")
    pil_end = time.time()
    print(f"pil image open time {pil_end - pil_start}")
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
    
    print(f"binary prediction {binary_prediction.shape}")

    stop_time = time.time()
    num_images_scored += 1
    print(f"Took {stop_time - start_time} to score {num_images_scored} image(s)") #:\n{image_path}")
    print(f"Took {(stop_time - start_time) / (num_images_scored)} seconds per image, or {(num_images_scored) / (stop_time - start_time)} images per second")

    score_image = PIL.Image.fromarray(
        np.clip(binary_prediction * 255.0, 0, 255).astype(np.uint8))

    # display(img_pil)
    # display(score_image)
    if (save_color_information_into_masks):
        create_pil_start = time.time()
        out_hwc_rgba_uint8 = np.zeros(shape=(height, width, 4), dtype=np.uint8)
        out_hwc_rgba_uint8[:, :, :3] = hwc_np_uint8
        out_hwc_rgba_uint8[:, :, 3] = binary_prediction * 255
        out_pil = PIL.Image.fromarray(out_hwc_rgba_uint8)
        create_pil_stop = time.time()
        print(f"Create PIL took {create_pil_stop - create_pil_start}")
        # save it to file:

        out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
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
        out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
        png_save_start = time.time()
        out_pil.save(out_file_name, "PNG")  # fast because it is black white
        png_save_stop = time.time()
        print(f"Took {png_save_stop - png_save_start} seconds to save png")

    # img_crop = img_pil.crop((i_min, j_min, i_max, j_max))
#     img_crop.save(cropped_image_path)
        
    print(f"pri {out_file_name}")

# if True:
#     plt.figure(figsize=(16, 9))
#     plt.imshow(out_pil)
#     plt.figure(figsize=(16, 9))
#     plt.imshow(img_crop)
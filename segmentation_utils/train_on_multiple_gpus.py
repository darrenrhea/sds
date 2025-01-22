"""
Might want to delete:
find ~/gsw/400x400/ -type f -name '*.png' -delete
tmux new -s parallel
conda activate floor_not_floor
cd ~/r/segmentation_utils/
CUDA_VISIBLE_DEVICES=2,3 python -m fastai.launch train_on_multiple_gpus.py
"""

from all_imports_for_image_segmentation import *
from pathlib import Path
import PIL
from PIL import Image
from fastai.vision.all import (
    resnet34, unet_learner, SegmentationDataLoaders, aug_transforms
)
from fastai.distributed import *
import numpy as np
from annotated_data import get_list_of_annotated_images_from_several_directories
import time



crop_height = 400
crop_width = 400
regenerate_crops = True
desired_mask_names = ["nonfloor"]
target_mask = "nonfloor"

# # if downsampling, make sure to downsample the original training masks and point the function
# # get_list_of_annotated_images() to the directory containing the downsampled masks.
# list_of_annotated_images = get_list_of_annotated_images_from_several_directories(
#     must_have_these_masks=desired_mask_names,
#     directories_to_gather_from_with_limits = [
#         # (
#         #     Path(f"~/r/final_gsw1/GSWvBOS_06-05-2022_PGM_ESP_MXF").expanduser(),
#         #     1
#         # ),
#         # (
#         #     Path(f"~/r/final_gsw1/nonfloor_convention").expanduser(),
#         #     1
#         # ),
#         (
#             Path(f"~/r/brooklyn_nets_barclays_center/nonfloor_segmentation_train").expanduser(),
#             1000000
#         )
#     ]
# )

# num_frames = len(list_of_annotated_images)

# pp.pprint(list_of_annotated_images)

# print(f"num_frames = {num_frames}")

# # croppings_dir = "~/gsw/400x400"
# croppings_dir = "~/brooklyn_nets_barclays_center/fastai_vs_deeplabv3_400x400"
# Path(croppings_dir).expanduser().mkdir(parents=True, exist_ok=True)
# start = time.time()
# if regenerate_crops:
#     cropped_hand_annotated_training_data = get_numpy_arrays_of_croppings_and_their_masks(
#         list_of_annotated_images=list_of_annotated_images,
#         crop_height=crop_height,
#         crop_width=crop_width,
#         desired_mask_names=desired_mask_names,
#         mask_name_to_min_amount=dict(nonfloor=1000),
#         mask_name_to_max_amount=dict(nonfloor=crop_height*crop_width - 1),
#         how_many_originals=len(list_of_annotated_images),
#         how_many_crops_per_original=1000,
#         mask_encoding_type="alpha_channel_of_rgba"
#     )

#     dct = cropped_hand_annotated_training_data
#     print([k for k in dct.keys()])
#     assert dct["num_croppings_cut"] > 0
#     assert isinstance(dct["mask_name_to_cropped_masks"], dict)
#     assert isinstance(dct["cropped_originals"], np.ndarray)
#     for mask_name in desired_mask_names:
#     #     print(f"We have the masks known as {mask_name}")
#         assert dct["mask_name_to_cropped_masks"][mask_name].shape[0] == dct["cropped_originals"].shape[0]
#     for k in range(dct["num_croppings_cut"]):
        
#         save_hwc_np_uint8_to_image_path(dct["cropped_originals"][k], Path(f"{croppings_dir}/{k}_color.png").expanduser())

#         save_hwc_np_uint8_to_image_path(
#             dct["mask_name_to_cropped_masks"][target_mask][k],
#             Path(f"{croppings_dir}/{k}_{target_mask}.png").expanduser())

#         if (k%1000 == 0):
#             print(f"saving {k}")
#             update = time.time()
#             print(f"{update - start} seconds")
#     end = time.time()
#     print(f"total {end - start} seconds")

croppings_dir = "~/brooklyn_nets_barclays_center/fastai_vs_deeplabv3_400x400"
croppings_path = Path(croppings_dir).expanduser()
num_crops_per_frame = 1000

fnames = list(croppings_path.glob('*_color.png')) # list of input frame paths
num_frames = int(len(fnames)/num_crops_per_frame)

# print(f"num frames = {num_frames}")
def label_func(fn):
    return croppings_path/f"{fn.stem[:-6]}_{target_mask}{fn.suffix}"

len([label_func(fn) for fn in fnames])

codes = np.array(["background", "foreground"])

img_pil = Image.open(label_func(fnames[3]))
np_img = np.array(img_pil)
np_img.shape

dls = SegmentationDataLoaders.from_label_func(
    croppings_path, 
    bs=32, 
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

train_dataloader, test_dataloader = dls
print("The type of train_dataloader is:")
print(type(train_dataloader))
print("The type of test_dataloader is:")
print(type(test_dataloader))

xb, yb = train_dataloader.one_batch()
print(type(xb))
print(type(yb))
print(f"xb.size() = {xb.size()}")
print(f"yb.size() = {yb.size()}")


# lets resume training from the PHI model:

# num_epochs = 6000
num_epochs = 32
model_path = Path("~/r/trained_models").expanduser()
model = unet_learner(dls, resnet34)
# model = model.load(
#     Path("~/r/trained_models/gsw_finals_400p_400p_res34_50e_24f_full_res").expanduser())

with model.distrib_ctx(sync_bn=False):  # if you don't put in this is will say some whack about batch norm
    for i in range(1, num_epochs + 1):
        model.fine_tune(1)
        # model_name = f"crap_gsw_finals_400p_400p_res34_{i}e_{num_frames}f_full_res"
        model_name = f"brooklyn_fastai_vs_deeplabv3_400p_400p_res34_{i}e_{num_frames}f"
        print(f"Saving model {model_name}")
        model.save(model_path / model_name)
        # this fails!? Maybe wait a bit
        # assert Path(model_path/f"{model_name}.pth").expanduser().exists()


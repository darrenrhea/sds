# ---
# jupyter:
#   jupytext:
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Environment (conda_floor_not_floor)
#     language: python
#     name: conda_floor_not_floor
# ---

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
from all_imports_for_image_segmentation import *
from pathlib import Path
import albumentations as A
import cv2
from image_displayers_for_jupyter import display_numpy_hw_grayscale_image
import PIL
from PIL import Image
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
from image_openers import open_a_grayscale_png_barfing_if_it_is_not_grayscale

torch.cuda.set_device('cuda:2')


def display_np_hw_0or1_image(x : np.ndarray):
    """
    Given a numpy matrix filled full of zeroes and ones,
    display it graphically as a black (0) and white (1) image within Jupyter Notebook.
    
    This is a generally useful procedure for displaying in Jupyter Notebook
    a binary (0 or 1 are the only allowed values)
    numpy 2D array under convention height x width
    """
    print("Running display_np_hw_0or1_image")
    assert x.dtype == np.uint8
    assert x.ndim == 2
    assert np.min(x) >= 0
    assert np.max(x) <= 1
    assert np.min(x) != np.max(x), "Warning constant image"
    x_uint8 = (x > 0).astype("uint8") * 255
    display(PIL.Image.fromarray(x_uint8))


use_case_name = "brooklyn"  # ultimately we would like this to generalize to many sports.  We have a long way to go on this.
data_dir = Path("~/r/brooklyn_nets_barclays_center/nonfloor_segmentation_downsampled_one_third/").expanduser()
# crop_height = 280  # we cut slightly bigger rectangles out so that we can wiggle it to promote translation invariance
# crop_width = 320
crop_height = 224  # we cut slightly bigger rectangles out so that we can wiggle it to promote translation invariance
crop_width = crop_height  # we crop out squares
desired_mask_names = ["nonfloor", "inbounds"]  # the "relevant" mask tells the loss function whether we care about accuracy for each particular pixel
regenerate_crops = True

list_of_annotated_images = get_list_of_annotated_images(
    use_case_name=use_case_name,
    must_have_these_masks=desired_mask_names
)

pp.pprint(list_of_annotated_images)

# +
dct = cut_this_many_interesting_subrectangles_from_annotated_image(
    annotated_image=list_of_annotated_images[0],  # which annotated_image to cut croppings from
    how_many_croppings_to_cut_out=1,
    crop_height=crop_height,
    crop_width=crop_width,
    desired_mask_names=desired_mask_names,
    mask_name_to_min_amount=dict(inbounds=1000, nonfloor=1000),  # we are forcing croppings to have at least one pixel of mainrectangle mask on
    mask_name_to_max_amount=dict(),
    mask_encoding_type="alpha_channel_of_rgba",
)
assert dct["success"]

for k in range(dct["num_croppings"]):
    print("original:")
    display(
        PIL.Image.fromarray(
            dct["cropped_originals"][k]
        )
    )
    for mask_name in desired_mask_names:
        print(f"{mask_name}:")
        display(
        PIL.Image.fromarray(
            255 * dct["mask_name_to_cropped_masks"][mask_name][k]
        )
    )
# -

croppings_dir = "~/r/brooklyn_nets_barclays_center/224x224_one_third_downsampled_croppings"
if regenerate_crops:
    cropped_hand_annotated_training_data = get_numpy_arrays_of_croppings_and_their_masks(
        list_of_annotated_images=list_of_annotated_images,
        crop_height=crop_height,
        crop_width=crop_width,
        desired_mask_names=desired_mask_names,
        mask_name_to_min_amount=dict(inbounds=1000, nonfloor=1000),
        mask_name_to_max_amount=dict(),
        how_many_originals=len(list_of_annotated_images),
        how_many_crops_per_original=1000,
        mask_encoding_type="alpha_channel_of_rgba"
    )

    dct = cropped_hand_annotated_training_data
    print([k for k in dct.keys()])
    assert dct["num_croppings_cut"] > 0
    assert isinstance(dct["mask_name_to_cropped_masks"], dict)
    assert isinstance(dct["cropped_originals"], np.ndarray)
    for mask_name in desired_mask_names:
    #     print(f"We have the masks known as {mask_name}")
        assert dct["mask_name_to_cropped_masks"][mask_name].shape[0] == dct["cropped_originals"].shape[0]
    for k in range(dct["num_croppings_cut"]):
        print(f"saving {k}")
    #     print("saving original:")
        save_hwc_np_uint8_to_image_path(dct["cropped_originals"][k], Path(f"{croppings_dir}/{k}_color.png").expanduser())
    #     display(
    #         PIL.Image.fromarray(
    #             dct["cropped_originals"][k]
    #         )
    #     )

    #     print(f"saving {mask_name}:")
        save_hwc_np_uint8_to_image_path(dct["mask_name_to_cropped_masks"]["nonfloor"][k], Path(f"{croppings_dir}/{k}_nonfloor.png").expanduser())
    #         display(
    #         PIL.Image.fromarray(
    #             255 * dct["mask_name_to_cropped_masks"][mask_name][k]
    #         )
    #     )

from pathlib import Path
croppings_path = Path(croppings_dir).expanduser()

fnames = list(croppings_path.glob('*_color.png')) # list of input frame paths


def label_func(fn): return croppings_path/f"{fn.stem[:-6]}_nonfloor{fn.suffix}"
# label_func(fnames[0])
# for i in range(len(fnames)):
#     print(label_func(fnames[i]))
len([label_func(fn) for fn in fnames])

# +
# print(fnames[0:10])

# +
# list_of_ids = [fn.stem[:-6] for fn in fnames]
# print(list_of_ids[0:10])

# +
# train_test_splitter = RandomSplitter(valid_pct=0.34, seed=3412)
# train_test_split = train_test_splitter(list_of_ids)
# print(train_test_split)

# +
# def primary_key_to_image_path(o: str) -> Path:
#     return croppings_path / f"{o}_color.png"

# primary_key_to_label_path = lambda o: croppings_path / f"{o}_nonfloor.png"

# +
# def get_hw_0or1_np_uint8_from_alpha_channel_of_RGBA(image_path : Path) -> np.ndarray:
    
#     pil_image = PIL.Image.open(str(image_path))
#     assert pil_image.mode == "RGBA", f"{image_path} not an RGBA image"
#     hw_0or1_np_uint8 = np.array(pil_image)[:, :, 3]  # slurp out the alpha channel only, the 3ith channel
#     # the alpha channel is 0or255 so we can see it via pri, but now we want it in the 0or1 standard
#     hw_0or1_np_uint8 = (hw_0or1_np_uint8 > 128).astype(np.uint8)
#     assert np.max(hw_0or1_np_uint8) <= 1  # make sure it is classic binary, i.e. 0or1
#     assert hw_0or1_np_uint8.dtype == np.uint8
#     return hw_0or1_np_uint8

# +
# def extract_A_from_RGBA_to_PILMask(path_to_RGBA_image : Path) -> PILMask:
#     """
#     We have RGBA PNGs you can look at with pri,
#     with the mask stored in the alpha channel as either 0 or 255,
#     but we need a fastai.vision.core.PILMask
#     """
#     msk_np = get_hw_0or1_np_uint8_from_alpha_channel_of_RGBA(
#         image_path=path_to_RGBA_image
#     )
#     return PILMask.create(msk_np)

# +
# an_id = list_of_ids[1]
# img_path = primary_key_to_image_path(an_id)
# print(img_path)
# corresponding_mask_path = primary_key_to_label_path(an_id)
# print(corresponding_mask_path)
# img = PILImage.create(img_path)
# print(img.shape)
# print(type(img))
# mask = open_a_grayscale_png_barfing_if_it_is_not_grayscale(corresponding_mask_path)
# print(mask.shape)
# print(type(mask))

# +
# img.show()

# +
# display_numpy_hw_grayscale_image(mask)

# +
# tfms = [
#     [
#         primary_key_to_image_path,
#         PILImage.create
#     ],
#     [
#         primary_key_to_label_path,
#         open_a_grayscale_png_barfing_if_it_is_not_grayscale
#     ]
# ]

# +
# cv_dsets = Datasets(items=list_of_ids, tfms=tfms, splits=train_test_split)
# print(type(cv_dsets))
# print(cv_dsets[0])
# print(cv_dsets[5])

# +
# class MessWithForegroundColorOnlyTransform(ItemTransform):
#     """
#     do not feed this thing any augmenter aug that is "spatial", like rotation, flipping etc.
#     We are assuming it does not alter the mask.
#     """
#     split_idx = 0  # they have some magic to only apply it to training data if that is your thing.
    
#     def __init__(self, aug):
#         """
#         Takes in a data-augmenter aug,
#         like A.ColorJitter(hue=0.9, always_apply=True, p=1.0) or something.
#         Does that augmentation, but in a way that protects the background from changing color.
#         """
#         self.aug = aug
    
#     def encodes(self, x):
#         img, mask = x
#         image_np = np.array(img)
#         mask_np = np.array(mask)
#         # albumentations does its thing:
#         augmented = self.aug(image=image_np, mask=mask_np)  # apply the augmenter, which screws with everyone
#         augmented_image_np = augmented["image"]
        
#         augmented_image_np[mask_np == 0] = image_np[mask_np == 0]  # restore the background to the original color
#         return PILImage.create(augmented_image_np), PILMask.create(mask_np)

# +
# aug_tfms = aug_transforms(
#         mult=1.0,
#         do_flip=True, 
#         flip_vert=False, 
#         max_rotate=10.0, 
#         min_zoom=1.0, 
#         max_zoom=1.1, 
#         max_lighting=0.2, 
#         max_warp=0.2, 
#         p_affine=0.75, 
#         p_lighting=0.75, 
#         xtra_tfms=None, 
#         size=None, 
#         mode='bilinear', 
#         pad_mode='reflection', 
#         align_corners=True, 
#         batch=False, 
#         min_scale=1.0
#     )
# -

codes = np.array(["background", "foreground"])

# +
# dls = cv_dsets.dataloaders(
#     bs=32,
#     after_item=[
#         ToTensor(),
#         IntToFloatTensor(),
#         MessWithForegroundColorOnlyTransform(
#             aug=A.ColorJitter(hue=0.9, brightness=0.6, always_apply=True, p=1.0)
#         )
#     ],
#     after_batch=[*aug_tfms]
# )

# +
# train_dataloader, test_dataloader = dls
# print("The type of train_dataloader is:")
# print(type(train_dataloader))
# print("The type of test_dataloader is:")
# print(type(test_dataloader))

# +
# xb, yb = train_dataloader.one_batch()
# print(type(xb))
# print(type(yb))
# print(f"xb.size() = {xb.size()}")
# print(f"yb.size() = {yb.size()}")

# +
# train_dataloader.show_batch(max_n=10, figsize=(13, 8))

# +
# for i in range(32):
#     display_numpy_hwc_rgb_image(
#         np.array(
#             xb.cpu()[i,:,:,:]
#         ).transpose(1,2,0)
#     )
# -

img_pil = Image.open(label_func(fnames[3]))
np_img = np.array(img_pil)
np_img.shape

# !nvidia-smi

# +
# dls = SegmentationDataLoaders.from_label_func(
#     croppings_path, 
#     bs=32, 
#     fnames = fnames, 
#     label_func = label_func, 
#     codes = codes,
#     valid_pct=0.1,
#     seed=42, # random seed,
#     item_tfms=[
#         ToTensor(),
#         IntToFloatTensor(),
#         MessWithForegroundColorOnlyTransform(
#             aug=A.ColorJitter(hue=0.9, brightness=0.6, always_apply=True, p=1.0)
#         )
#     ],
#     batch_tfms=aug_transforms(
#         mult=1.0,
#         do_flip=True, 
#         flip_vert=False, 
#         max_rotate=10.0, 
#         min_zoom=1.0, 
#         max_zoom=1.1, 
#         max_lighting=0.2, 
#         max_warp=0.2, 
#         p_affine=0.75, 
#         p_lighting=0.75, 
#         xtra_tfms=None, 
#         size=None, 
#         mode='bilinear', 
#         pad_mode='reflection', 
#         align_corners=True, 
#         batch=False, 
#         min_scale=1.0)
# )
# -

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

for i in range(32):
    display_numpy_hwc_rgb_image(
        np.array(
            xb.cpu()[i,:,:,:]
        ).transpose(1,2,0)
    )

model = unet_learner(dls, resnet34)
# model = model.load(Path("~/r/trained_models/fastai_23e_50f").expanduser())

# +
# get_c(dls)
# -

model.fine_tune(1)

model_path = Path("~/r/trained_models").expanduser()
model_name = "brooklyn_224p_224p_res34_1e_22f_downsampled_one_third"

model.save(model_path/model_name)
assert Path(model_path/f"{model_name}.pth").expanduser().exists()

num_epochs = 32
model_path = Path("~/r/trained_models").expanduser()
for i in range(2, num_epochs + 1):
    model.fine_tune(1)
    model_name = f"brooklyn_224p_224p_res34_{i}e_22f_downsampled_one_third"
    print(f"Saving model {model_name}")
    model.save(model_path/model_name)
    assert Path(model_path/f"{model_name}.pth").expanduser().exists()

num_epochs = 100
model_path = Path("~/r/trained_models").expanduser()
for i in range(33, 33 + num_epochs + 1 ):
    model.fine_tune(1)
    model_name = f"fastai_brooklyn_224p_224p_res34_{i}e_16f"
    print(f"Saving model {model_name}")
    model.save(model_path/model_name)
    assert Path(model_path/f"{model_name}.pth").expanduser().exists()

# +
# num_epochs = 32
# model_path = Path("~/r/trained_models").expanduser()
# for i in range(17, num_epochs + 1):
#     model.fine_tune(1)
#     model_name = f"fastai_resnet34_224p_{i}e_10f"
#     print(f"Saving model {model_name}")
#     model.save(model_path/model_name)
#     assert Path(model_path/f"{model_name}.pth").expanduser().exists()

# +
# model.fine_tune(20)

# +
# learn = unet_learner(dls, resnet34)
# learn.fine_tune(20)
# -

from pathlib import Path
from fastai.vision.all import *
model_path = Path("~/r/trained_models").expanduser()
model_name = "fastai_224p_res34_32e_10f"

from fastai.vision.all import *
import numpy as np
croppings_dir = "~/r/gsw1/224_croppings"
path = Path(croppings_dir).expanduser()
fnames = list(path.glob('*_color.png')) # list of input frame paths
def label_func(fn): return path/f"{fn.stem[:-6]}_nonfloor{fn.suffix}"
codes = np.array(["foreground", "background"])
dls = SegmentationDataLoaders.from_label_func(
    path, 
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
model = unet_learner(dls, resnet34)
model = model.load(model_path/model_name)

# +
# model.summary()

# +
# save_model = True
# if save_model:
#     model_path = model_path
#     model_name = model_name
#     learn.save(model_path/model_name)
#     assert Path(model_path/f"{model_name}.pth").expanduser().exists()

# +
# model = unet_learner(dls, resnet34)
# model = model.load(model_path/model_name)

# +
# model.fine_tune(16)

# +
# model = unet_learner(dls, resnet34)
# model = model.load(Path("~/r/trained_models/fastai_32e_14f").expanduser())
# model.fine_tune(125)
# -

predict_image = fnames[2]

import matplotlib.pyplot as  pp
import time
start_time = time.time()
a = learn.predict(predict_image)
end_time = time.time()
print(f"predict time {end_time - start_time}")
pp.imshow(a[0])

pil_im = Image.open(label_func(predict_image), 'r')
display(PIL.Image.fromarray(255 * np.array(pil_im)))

original_color = Image.open(predict_image)
display(original_color)

# +
# type(learn)

# +
# model.summary()

# +
# model.save(Path("~/r/trained_models/fastai_157e_14f").expanduser())

# +
# learn.show_results(max_n=2, figsize=(20,20))

# +
# torch_device = my_pytorch_utils.get_the_correct_gpu("Tesla", which_copy=0)

# +
# possibly after loading it full of weights, you got to move the model onto the gpu or else bad things
# about Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
# will happen:
# model = model.to(torch_device)  # stuff it onto the gpu

# +
# from stitch_togetherer import stitch_togetherer
# import numpy as np
# import torch
# import torch.nn.functional as F
# import time
# from fastai.vision.all import *
# threshold = 0.5
# input_extension = "jpg" # bmp is faster to read and write, but huge
# video_name = "gsw1"
# masking_attempt_id = "fastai_2"
# save_color_information_into_masks = True
# # # height = 1080
# # # width = 1920

# i_min = 0
# j_min = 0
# j_max = 1920
# i_max = 1080
# width = j_max - j_min
# height = i_max - i_min

# masking_attempts_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts").expanduser()
# masking_attempts_dir.mkdir(exist_ok=True)
# out_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts/{masking_attempt_id}").expanduser()
# out_dir.mkdir(exist_ok=True)
# start_time = time.time()
# num_images_scored = 0
# for frame_index in range(160000, 300000, 1000):
#     color_original_pngs_dir = Path(
#         f"~/awecom/data/clips/{video_name}/frames").expanduser()
#     image_path = color_original_pngs_dir / f"{video_name}_{frame_index:06d}.{input_extension}"
#     cropped_image_path = color_original_pngs_dir / f"{video_name}_{frame_index:06d}_cropped.{input_extension}"
#     assert image_path.is_file(), f"{image_path} does not exist!"
#     img_pil = PIL.Image.open(str(image_path)).convert("RGB")
#     hwc_np_uint8 = np.array(img_pil)
    
#     binary_prediction = stitch_togetherer(
#         hwc_np_uint8=hwc_np_uint8,
#         i_min=i_min,
#         i_max=i_max,
#         j_min=j_min,
#         j_max=j_max,
#         torch_device=torch_device,
#         model=model,
#         threshold=threshold,
#         stride=30,
#         batch_size=2
#     )
    
#     print(f"binary prediction {binary_prediction.shape}")

#     stop_time = time.time()
#     num_images_scored += 1
#     print(f"Took {stop_time - start_time} to score {num_images_scored} images:\n{image_path}")
#     print(f"Took {(stop_time - start_time) / (num_images_scored)} seconds per image, or {(num_images_scored) / (stop_time - start_time)} images per second")

#     score_image = PIL.Image.fromarray(
#         np.clip(binary_prediction * 255.0, 0, 255).astype(np.uint8))

#     # display(img_pil)
#     # display(score_image)
#     if (save_color_information_into_masks):
#         out_hwc_rgba_uint8 = np.zeros(shape=(height, width, 4), dtype=np.uint8)
#         out_hwc_rgba_uint8[:, :, :3] = hwc_np_uint8
#         out_hwc_rgba_uint8[:, :, 3] = binary_prediction * 255
#         out_pil = PIL.Image.fromarray(out_hwc_rgba_uint8)
#         # save it to file:

#         out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
#         out_pil.save(out_file_name, "PNG")
#     else:
#         out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
#         out_hw_grayscale_uint8[:, :] = binary_prediction * 255
#         out_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
#         # save it to file:
#         out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
#         out_pil.save(out_file_name, "PNG")
    
#     img_crop = img_pil.crop((i_min, j_min, i_max, j_max))
# #     img_crop.save(cropped_image_path)
        
#     print(f"See {out_file_name}")

# if True:
#     plt.figure(figsize=(16, 9))
#     plt.imshow(out_pil)
#     plt.figure(figsize=(16, 9))
#     plt.imshow(img_crop)

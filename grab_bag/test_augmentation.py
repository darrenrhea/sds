from pathlib import Path
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
"""
ImageMaskMotionBlur only works if the masks are dtype float with range [0,1]
Maybe it can be changed to accept masks that are uint8?
"""
from print_image_in_iterm2 import print_image_in_iterm2
import cv2
import numpy as np

import albumentations as A

from albumentations.core.transforms_interface import DualTransform

from imagemasktransform import ImageMaskMotionBlur, ImageMaskBlur
from get_training_augmentation import get_training_augmentation


image_set_name = "flatled"

if image_set_name == "baseball":
    # give it an rgb original frame:
    fn_frame ="fixtures/20200725PIT-STL-CFCAM-PITCHCAST_inning1_000600.jpg"

    # give it the labeling target mask that is either rgba where we only use the alpha channel or a grayscale mask:
    fn_mask ="fixtures/20200725PIT-STL-CFCAM-PITCHCAST_inning1_000600_nonfloor.png"

    # importance weights mask:
    fn_weights = "fixtures/20200725PIT-STL-CFCAM-PITCHCAST_inning1_000600_relevance.png"

elif image_set_name == "led":
    # give it an rgb original frame:
    fn_frame ="fixtures/hou-lac-2023-11-14_222073_nonfloor.png"

    # give it an mask that is either rgba where we only use the alpha channel or a grayscale mask:
    fn_mask ="fixtures/hou-lac-2023-11-14_222073_nonfloor.png"

    # fn_weights = '20200725PIT-STL-CFCAM-PITCHCAST_inning1_000600_nonfloor.png'
    fn_weights = "fixtures/hou-lac-2023-11-14_222073_relevance.png"

elif image_set_name == "flatled":
    # give it an rgb original frame:
    fn_frame ="/shared/flattened_fake_game5/ripbased2/bos-mia-2024-04-21-mxf_548500_fake217837038218635_original.png"

    # give it an mask that is either rgba where we only use the alpha channel or a grayscale mask:
    fn_mask ="/shared/flattened_fake_game5/ripbased2/bos-mia-2024-04-21-mxf_548500_fake217837038218635_nonfloor.png"

    # importance weights mask:
    fn_weights = "/shared/flattened_fake_game5/ripbased2/bos-mia-2024-04-21-mxf_548500_fake217837038218635_relevance.png"
# are the masks dtype uint8 or float? 
# uint8_or_float = "uint8"
uint8_or_float = "float"

if uint8_or_float == "float":
    print("Trying with float")
    image = cv2.cvtColor(cv2.imread(fn_frame), cv2.COLOR_BGR2RGB)
    mask_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=Path(fn_mask).resolve()
    )
    mask = mask_u8.astype(float) / 255.
    weight = cv2.imread(fn_weights, -1)[:, :].astype(float) / 255.
elif uint8_or_float == "uint8":
    print("Trying with uint8")
    image = cv2.cvtColor(cv2.imread(fn_frame), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(fn_mask, -1)[:, :, 3]
    weight = cv2.imread(fn_weights, -1)[:, :]
else:
    raise ValueError(f"{uint8_or_float=} is not valid")

print(f"{type(image)=}")
print(f"{image.shape=}")
print(f"{image.dtype=}")

print(f"{type(mask)=}")
print(f"{mask.dtype=}")
print(f"{mask.shape=}")

print(f"{type(weight)=}")
print(f"{weight.shape=}")
print(f"{weight.dtype=}")

augmentation = get_training_augmentation(
    augmentation_id="forflat",
    frame_width=image.shape[1],
    frame_height=image.shape[0],
)



data = {"image": image, "mask": mask, "importance_mask": weight}

for _ in range(10):
        
    augmented = augmentation(**data)
    image_augmented = augmented["image"]
    mask_augmented = augmented["mask"]
    weight_augmented = augmented["importance_mask"]




    if uint8_or_float == "uint8":
        assert weight_augmented.dtype == np.uint8, f"{type(weight_augmented.dtype)=}"
        assert mask_augmented.dtype == np.uint8
        print("image_augmented:")
        print_image_in_iterm2(rgb_np_uint8 = image_augmented)
        print("mask_augmented:")
        print_image_in_iterm2(grayscale_np_uint8 = mask_augmented)
        print("weight_augmented:")
        print_image_in_iterm2(grayscale_np_uint8 = weight_augmented)
    else:
        print("image_augmented:")
        print_image_in_iterm2(rgb_np_uint8 = image_augmented)
        print("mask_augmented:")
        print_image_in_iterm2(
            grayscale_np_uint8 = np.round(mask_augmented * 255).astype(np.uint8).astype(np.uint8)
        )
        print("weight_augmented:")
        print_image_in_iterm2(
            grayscale_np_uint8 = np.round(weight_augmented * 255).astype(np.uint8)
        )


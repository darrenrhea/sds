import numpy as np
from print_image_in_iterm2 import print_image_in_iterm2
from blackpad_preprocessor import blackpad_preprocessor
from open_mask_image import open_mask_image
from open_just_the_rgb_part_of_image import open_just_the_rgb_part_of_image
from pathlib import Path


def test_blackpad_preprocessor():

    original_path = Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526.jpg").expanduser().resolve()
    target_mask_path = Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526_nonfloor.png").expanduser().resolve()
    weight_mask_path = Path("~/alpha_mattes_temp/hou-lac-2023-11-14_243526_relevance.png").expanduser().resolve()
    
    print(f"{original_path=}")
    print(f"{target_mask_path=}")
    print(f"{weight_mask_path=}")

    desired_width = 1920
    desired_height = 1088

    params = dict(
        desired_width=desired_width,
        desired_height=desired_height
    )

    raw_original = open_just_the_rgb_part_of_image(image_path=original_path)
    raw_target_mask = open_mask_image(mask_path=target_mask_path)
    raw_weight_mask = open_mask_image(mask_path=weight_mask_path)
    print(f"before preprocessing:")
    print_image_in_iterm2(rgb_np_uint8=raw_original)
    print_image_in_iterm2(grayscale_np_uint8=raw_target_mask)
    print_image_in_iterm2(grayscale_np_uint8=raw_weight_mask)

    raw_stack_of_channels = np.concatenate(
        (
            raw_original,
            raw_target_mask[:, :, np.newaxis],
            raw_weight_mask[:, :, np.newaxis]
        ),
        axis=2
    )

    channel_stack = blackpad_preprocessor(
        channel_stack=raw_stack_of_channels,
        params=params
    )

    image = channel_stack[:, :, :3]
    target_mask = channel_stack[:, :, 3]
    weight_mask = channel_stack[:, :, 4]

    assert isinstance(image, np.ndarray)
    assert image.shape[0] == desired_height
    assert image.shape[1] == desired_width
    assert image.shape[2] == 3

    assert isinstance(target_mask, np.ndarray)
    assert image.shape[0] == target_mask.shape[0]
    assert image.shape[1] == target_mask.shape[1]
    assert target_mask.ndim == 2

    assert isinstance(weight_mask, np.ndarray)
    assert image.shape[0] == weight_mask.shape[0]
    assert image.shape[1] == weight_mask.shape[1]
    assert weight_mask.ndim == 2

    print(f"after preprocessing:")
    print(f"{image.shape=}, {target_mask.shape=}, {target_mask.shape=}")
    print_image_in_iterm2(rgb_np_uint8=image)
    print_image_in_iterm2(grayscale_np_uint8=target_mask)
    print_image_in_iterm2(grayscale_np_uint8=weight_mask)
        

if __name__ == "__main__":
    test_blackpad_preprocessor()






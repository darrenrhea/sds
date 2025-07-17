from prii import (
     prii
)
from cutout_a_patch_from_fullsize import (
     cutout_a_patch_from_fullsize
)
import numpy as np
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
import PIL.Image



def test_cutout_a_patch_from_fullsize():
    np.set_printoptions(threshold=8000)
    num_patches_to_generate = 10
    patch_width = 512
    patch_height = 256
    fullsize_image_path = Path("/shared/flattened_training_data/summer_league_2024/slday8game1_318000_original.jpg").resolve()
    fullsize_mask_path = Path("/shared/flattened_training_data/summer_league_2024/slday8game1_318000_nonfloor.png").resolve()
    fullsize_relevance_mask_path = Path("/shared/flattened_training_data/summer_league_2024/slday8game1_318000_relevance.png").resolve()

    assert fullsize_image_path.is_file(), f"{fullsize_image_path} is not a file"
    fullsize_rgb_np_u8 = np.array(PIL.Image.open(fullsize_image_path))
    fullsize_mask_np_u8 = np.array(PIL.Image.open(fullsize_mask_path))
    fullsize_relevance_mask_np_u8 = np.array(PIL.Image.open(fullsize_relevance_mask_path))

    fullsize_image_np_u8 = np.zeros(
        shape=(fullsize_rgb_np_u8.shape[0], fullsize_rgb_np_u8.shape[1], 5),
        dtype=np.uint8
    )
    fullsize_image_np_u8[:, :, 0:3] = fullsize_rgb_np_u8
    fullsize_image_np_u8[:, :, 3] = fullsize_mask_np_u8[:, :]
    fullsize_image_np_u8[:, :, 4] = fullsize_relevance_mask_np_u8[:, :]

    print("fullsize:")
    prii(fullsize_image_np_u8[:, :, :4])

    patch = cutout_a_patch_from_fullsize(
        fullsize_image_np_u8=fullsize_image_np_u8,
        patch_width=patch_width,
        patch_height=patch_height,
    )
    assert patch.shape == (patch_height, patch_width, fullsize_image_np_u8.shape[2])


    print_out_some_examples = True
    if print_out_some_examples:
        print_image_in_iterm2(rgb_np_uint8=patch[:, :, :3])
        print_image_in_iterm2(rgba_np_uint8=patch[:, :, :4])
        print_image_in_iterm2(grayscale_np_uint8=patch[:, :, 3])
        print_image_in_iterm2(grayscale_np_uint8=patch[:, :, 4])


if __name__ == "__main__":
    test_cutout_a_patch_from_fullsize()
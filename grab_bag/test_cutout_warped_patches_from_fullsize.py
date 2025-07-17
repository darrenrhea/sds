
import time
import numpy as np
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
import PIL.Image
from cutout_warped_patches_from_fullsize import cutout_warped_patches_from_fullsize


def test_cutout_warped_patches_from_fullsize():
    np.set_printoptions(threshold=8000)
    num_patches_to_generate = 10
    patch_width = 700
    patch_height = 384
    fullsize_image_path = Path("fixtures/monaco_000018220.jpg").resolve()
    fullsize_mask_path = Path("fixtures/monaco_000018220_nonfloor.png").resolve()

    assert fullsize_image_path.is_file(), f"{fullsize_image_path} is not a file"
    fullsize_rgb_np_u8 = np.array(PIL.Image.open(fullsize_image_path))
    fullsize_mask_np_u8 = np.array(PIL.Image.open(fullsize_mask_path))

    fullsize_image_np_u8 = np.zeros(
        shape=(fullsize_rgb_np_u8.shape[0], fullsize_rgb_np_u8.shape[1], 4),
        dtype=np.uint8
    )
    fullsize_image_np_u8[:, :, 0:3] = fullsize_rgb_np_u8
    fullsize_image_np_u8[:, :, 3] = fullsize_mask_np_u8[:, :, 3]

    print_image_in_iterm2(rgba_np_uint8=fullsize_image_np_u8)

    start_time = time.time()
    patches = cutout_warped_patches_from_fullsize(
        fullsize_image_np_u8=fullsize_image_np_u8,
        patch_width=patch_width,
        patch_height=patch_height,
        num_patches_to_generate=num_patches_to_generate
    )
    stop_time = time.time()

    patches_per_second = num_patches_to_generate / (stop_time - start_time)

    print_out_some_examples = True
    if print_out_some_examples:
        for i in range(num_patches_to_generate):
            patch = patches[i, ...]
            print_image_in_iterm2(rgb_np_uint8=patch[:, :, 0:3])
            print_image_in_iterm2(rgba_np_uint8=patch)
            print_image_in_iterm2(grayscale_np_uint8=patch[:, :, 3])

    print(f"{patches_per_second=}")

if __name__ == "__main__":
    test_cutout_warped_patches_from_fullsize()
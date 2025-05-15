import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
from typing import Optional

from image_openers import (
    open_alpha_channel_image_as_a_single_channel_grayscale_image,
    open_as_grayscale_regardless
)
from write_rgb_hwc_np_u8_to_png import write_rgb_hwc_np_u8_to_png

def see_alpha_channel_differences(
    alpha_source_a_file_path: Path,  # the original image
    alpha_source_b_file_path: Path,  # the changed image
    rgb_source_file_path: Optional[Path],  # the original image
    save_file_path: Optional[Path],  # the diff visualization
    print_in_terminal: bool
):
    """
    Compares the alpha channel data of two images in iterm2.
    RGB data is ignored.
    """
    
    alpha_np_1 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
       alpha_source_a_file_path
    )
    alpha_np_2 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        alpha_source_b_file_path
    )
    
    if rgb_source_file_path is not None:
        grayscale_np = open_as_grayscale_regardless(rgb_source_file_path) // 2
        rgb_np = np.stack([grayscale_np, grayscale_np, grayscale_np], axis=2)
    else:
        rgb_np =np.zeros(
            shape=(alpha_np_1.shape[0], alpha_np_1.shape[1], 3),
            dtype=np.uint8
        )
    
    
    num_differences = np.sum(alpha_np_2 != alpha_np_1)
    print(f"Number of alpha differences: {num_differences}")
    assert alpha_np_1.shape == alpha_np_2.shape, "ERROR: Images are not the same size!"
    difference = alpha_np_2.astype(np.int32) - alpha_np_1.astype(np.int32)
    # if the difference is positive, something got added to the visible selection
    green = (difference).clip(0, 255).astype(np.uint8)
    red = (-difference).clip(0, 255).astype(np.uint8)

   
    rgb_np[:, :, 0] = (rgb_np[:, :, 0].astype(np.int32) + red).clip(0, 255).astype(np.uint8)
    rgb_np[:, :, 1] = (rgb_np[:, :, 1].astype(np.int32) + green).clip(0, 255).astype(np.uint8)

    final_pil = PIL.Image.fromarray(rgb_np)
    if print_in_terminal:
        print_image_in_iterm2(image_pil=final_pil)
    if save_file_path is not None:
        write_rgb_hwc_np_u8_to_png(rgb_np, save_file_path, verbose=True)



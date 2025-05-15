import sys
import PIL
import PIL.Image
import numpy as np
from typing import Optional
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
import argparse

from image_openers import (
    open_alpha_channel_image_as_a_single_channel_grayscale_image,
    open_image_as_rgb_np_uint8_ignoring_any_alpha
)

def pric(
    rgb_path: Path,
    alpha_path: Path,
    invert: bool,
    saveas: Optional[Path]
):
    """
    This is good for when the mask is correct in the alpha channel of one png, but
    """
    

    assert alpha_path.is_file(), f"No such file {alpha_path}"
    assert rgb_path.is_file(), f"No such file {rgb_path}"
    # print(
    #     f"Taking the alpha channel from {alpha_path} and the rgb info from {rgb_path} and combining them to form a colorized mask:"
    # )
    
    alpha_np_uint8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=alpha_path
    )

    if invert:
        alpha_np_uint8 = 255 - alpha_np_uint8


    rgb_np_uint8 = open_image_as_rgb_np_uint8_ignoring_any_alpha(
        abs_file_path=rgb_path
    )
   
    assert rgb_np_uint8.ndim == 3
    assert rgb_np_uint8.dtype == np.uint8
    assert rgb_np_uint8.shape[0] == alpha_np_uint8.shape[0]
    assert rgb_np_uint8.shape[1] == alpha_np_uint8.shape[1]
    assert rgb_np_uint8.shape[2] == 3
    combined_rgba_np_uint8 = np.zeros(
        shape=(alpha_np_uint8.shape[0], alpha_np_uint8.shape[1], 4),
        dtype=np.uint8,
    )
    combined_rgba_np_uint8[:, :, 3] = alpha_np_uint8
    combined_rgba_np_uint8[:, :, 0:3] = rgb_np_uint8
    combined_pil = PIL.Image.fromarray(combined_rgba_np_uint8)
    print_image_in_iterm2(
        image_pil=combined_pil,
    )
    if saveas is not None:
        assert isinstance(saveas, Path)
        combined_pil.save(str(saveas))


def pric_cli():
    argp = argparse.ArgumentParser()
    # positional arguments are automatically required:
    argp.add_argument("original_path", type=str)
    argp.add_argument("mask_path", type=str)
    argp.add_argument("--invert", action="store_true")
    argp.add_argument("--saveas", type=str, default=None)
    opt = argp.parse_args()
    original_path = Path(opt.original_path).resolve()
    mask_path = Path(opt.mask_path).resolve()
    assert original_path.is_file(), f"No such file {original_path}"
    assert mask_path.is_file(), f"No such file {mask_path}"
    if opt.saveas is not None:
        saveas = Path(opt.saveas).resolve()
    else:
        saveas = None
    pric(
        rgb_path=original_path,
        alpha_path=mask_path,
        invert=opt.invert,
        saveas=saveas
    )
    

if __name__ == "__main__":
    pric_cli()

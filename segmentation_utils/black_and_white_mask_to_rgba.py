import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
# from numpy.core.fromnumeric import searchsorted
from numpy.lib.npyio import save


def transfer_alpha_channel(alpha_path, rgb_path, save_path):
    """
    Sometimes the mask is expressed as a black and white png since they are
    very space-efficient,
    but we want to see it, so we put it onto the color image as
    and alpha channel rgb png.
    """
    print(
        f"Taking the alpha channel from {alpha_path} and the rgb info from {rgb_path} and combining them to form {save_path}"
    )
    alpha_pil = PIL.Image.open(str(alpha_path)).convert("L")  # the source of the good alpha channel
    alpha_np = np.array(alpha_pil)
    assert alpha_np.ndim == 2

    rgb_pil = PIL.Image.open(str(rgb_path)).convert(
        "RGBA"
    )  # the source of the rgb information. Even if it doesn't have an alpha channel, make one
    rgba_np = np.array(rgb_pil)
    rgba_np[:, :, 3] = alpha_np[:, :]
    save_pil = PIL.Image.fromarray(rgba_np)
    save_pil.save(save_path)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            """
Many of our foreground segmentation systems output segmentation as a
black-and-white/grayscale PNG where white=255 means foreground and black=0 means background.
This is because such black and white PNG files are quite small, merely tens of kilobytes,
and a typical video will have 100000 frames so the space problems would be serious if we did
otherwise.  

But human annotators that want to start with the machine's segmentation-attempt of a video frame
and then fix the mistakes that it made
will want to edit on an RGBA png where the color information is present so that
they can fix the mistakes by erasing and unerasing in GIMP, Photoshop, or whatever.
Such a full color information RGBA PNG is 3.2 Megabytes, but it is exactly what a
human annotator wants.

This program can take the b&w mask and the original color info and make
the combined RGBA png.

Usage:
   python black_and_white_mask_to_rbga.py <black_and_white_source_of_alpha.png> <source_of_original_rgb_info.jpg_or_png> <where_to_output_combined_rgba.png>
"""
        )
        sys.exit(1)

    alpha_path = Path(sys.argv[1])
    rgb_path = Path(sys.argv[2])
    save_path = Path(sys.argv[3])
    assert alpha_path.is_file(), f"No such file {alpha_path}"
    assert rgb_path.is_file(), f"No such file {rgb_path}"

    if save_path.is_file():
        ans = input(
            f"{save_path} already exists, are you sure you want to overwrite it? "
        )
        if ans not in ["yes", "Yes", "y", "Y"]:
            print("Stopping")
            sys.exit(1)

    transfer_alpha_channel(
        alpha_path=alpha_path, rgb_path=rgb_path, save_path=save_path
    )

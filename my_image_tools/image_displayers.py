"""
Sometimes you need to diff two jpegs or pngs.
"""
import sys
import PIL
import PIL.Image
import numpy as np
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2
from image_openers import open_a_grayscale_png_barfing_if_it_is_not_grayscale


def pri01grayscale(image_path):
    """
    We have grayscale PNGs whole gray values are 0 or 1 only,
    which is not perceiveable.
    """
    # cannot do these checks as they are, because it does not think of /dev/fd/63 as a file:

    # assert path1.is_file(), f"No such file {path1}"
    # assert path2.is_file(), f"No such file {path2}"

    # if save_path.is_file():
    #     ans = input(
    #         f"{save_path} already exists, are you sure you want to overwrite it? "
    #     )
    #     if ans not in ["yes", "Yes", "y", "Y"]:
    #         print("Stopping")
    #         sys.exit(1)
    image_np_uint8_bw_0or1 = open_a_grayscale_png_barfing_if_it_is_not_grayscale(
        image_path
    )

    final_pil = PIL.Image.fromarray(255 * image_np_uint8_bw_0or1)
    print_image_in_iterm2(image_pil=final_pil)


def mainpri01grayscale():
    """
    This is the "main" of the command line utility pridiff.
    """
    if len(sys.argv) < 2:
        print(
            """
Prints an grayscale-colorspaced PNG whose grayscale values only range in {0,1} in iterm2.

Usage:
   pri01grayscale image1.png 
"""
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])
    pri01grayscale(image_path)


if __name__ == "__main__":
    mainpri01grayscale()
